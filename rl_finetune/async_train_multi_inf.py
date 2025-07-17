import os
from dataclasses import asdict, dataclass
from datetime import timedelta
from functools import partial
from multiprocessing import Manager
from queue import Empty

import time

import pyrallis
import wandb
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp

from utils import init_pool
#from dataset_utils import IndexBuffer

from grpo_mm import generate_rollout_data, grpo_loss, compute_log_probs
from train_cadrille_grpo import TrainConfig, collate_img_pc_v1, get_reward_function, optimize_model_memory, setup, cleanup

from cad_recode_model_mm import Cadrille

from transformers import AutoProcessor
from dataset_utils import RealDatasetMM

from utils import init_pool

@dataclass
# class to hold IPC keys, that will be transferred between processes
class IPCKeys:
    INPUT_IDS: str = "input_ids"
    ATT_MASK: str = "attention_mask"
    COMP_MASK: str = "completion_mask"
    ADV: str = "advantages"
    OLD_LOGP: str  = "old_log_probs"    
    POINT_CLOUD: str = "point_cloud"
    IS_PC: str = "is_pc"
    IS_IMG: str = "is_img"
    PIXEL_VALUES_VIDEOS: str = "pixel_values_videos"
    VIDEO_GRID_THW: str = "video_grid_thw"
    AVG_REWARD: str = "avg_reward"
    LOGITS_TO_KEEP: str = "logits_to_keep"


def reward_inference_worker(queue, model, processor, train_data, config, rank):
    print("Initialized Multiprocesssing pool")
    
    init_pool(18)


    """GPU 0: sample rollouts, compute old log‑probs & advantages, enqueue minimal tensors."""
    torch.cuda.set_device(rank)


    sampler = DistributedSampler(train_data, num_replicas=config.num_reward_workers, rank=rank, shuffle=True)

    reward_fn = get_reward_function(config.failure_reward)
    step = 0

    dataloader = DataLoader(train_data, batch_size=config.batch_size // config.num_reward_workers, collate_fn=partial(collate_img_pc_v1, processor=processor, n_points=256), sampler=sampler,
                                num_workers=5, pin_memory=True)
    
    for epoch in range(config.train_epochs):

        print(f"Generator (Rank {rank}): Starting epoch {epoch + 1}/{config.train_epochs}.")
        sampler.set_epoch(epoch)

        for batch in dataloader:
            # synchronize the model parameters from GPU 1
            print(f"Generator (Rank {rank}): Synchronizing model parameters.")

            for param in model.parameters():
                dist.broadcast(param.data, src=config.num_reward_workers)

            print(f"Generating rollouts for batch {step + 1}/{len(dataloader)}")
            rollout, avg_reward = generate_rollout_data(
                model,
                reward_fn,
                processor,
                batch,
                config.num_generations,
                config.max_completion_length,
                top_samples=config.top_samples,
                gpg=config.use_gpg,
                buffer = None)
            
        
            payload = {} 
            for key in [
                IPCKeys.INPUT_IDS,
                IPCKeys.ATT_MASK,
                IPCKeys.COMP_MASK,
                IPCKeys.ADV,
                IPCKeys.OLD_LOGP,
                IPCKeys.POINT_CLOUD,
                IPCKeys.IS_PC,
                IPCKeys.IS_IMG,
                IPCKeys.LOGITS_TO_KEEP,
                IPCKeys.AVG_REWARD, 
                IPCKeys.PIXEL_VALUES_VIDEOS,
                IPCKeys.VIDEO_GRID_THW,

            ]:
                if key in rollout:
                    if isinstance(rollout[key], torch.Tensor):
                        payload[key] = rollout[key].detach().cpu().share_memory_()
                    else:
                        payload[key] = rollout[key]
            payload[IPCKeys.AVG_REWARD] = avg_reward
            queue.put(payload)
            step+=1

        # Signal to trainer that the epoch is finished
        queue.put(None)

    print(f"Generator (Rank {rank}): All epochs complete.")
    

def trainer_worker(queue, model, processor, config, rank):
    from torch.nn.utils.rnn import pad_sequence

    """GPU 1: compute loss, update model, evaluate."""
    print(f"Starting Trainer (Rank {rank}) ")
    torch.cuda.set_device(rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


    reward_function = get_reward_function(config.failure_reward)

    loss_fn = partial(grpo_loss, processor=processor,epsilon_high=config.epsilon_high, epsilon_low=config.epsilon_low, reward_function=reward_function)

    num_reward_workers = config.num_reward_workers
    
    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
    )

    os.environ["WANDB_RUN_ID"] = run.id
    
    # Initial broadcast of parameters
    for param in model.parameters():
        dist.broadcast(param.data, src=rank)

    step = 0
    optimizer.zero_grad()
    for epoch in range(config.train_epochs):

        print(f"Trainer (Rank {rank}): Starting epoch {epoch + 1}/{config.train_epochs}.")
        end_signals = 0

        while end_signals < num_reward_workers:
            mini_batches = []
            avg_rewards = []
            while len(mini_batches) != num_reward_workers:
                t0 = time.perf_counter()
                item = queue.get()
                wait = time.perf_counter() - t0 
                print(f"TIME to get sample from queue {wait}", flush=True)
                if item is None:
                    print(f"Trainer (Rank {rank}): Received end-of-epoch signal from one worker.", flush=True)
                    end_signals += 1
                    continue

                avg_rewards.append(item["avg_reward"])

                mini_batches.append(item)
            
            if not mini_batches:
                print(f"Trainer (Rank {rank}): Received {num_reward_workers} end-of-epoch signals.", flush=True)
                continue
            # compute the average reward across that concatenated batch
            avg_reward = sum(avg_rewards) / len(avg_rewards)

            # parameter updates following the direction of the loss
            for grpo_iter in range(config.batch_updates):
                t0 = time.perf_counter()
                optimizer.zero_grad()
                total_loss_in_iter = 0
                avg_reward = 0
                # we want 2 “mini‑batches” before we step
                for i in range(num_reward_workers):
                    # move tensors to GPU
                    rollout = {k: (v.to(rank) if isinstance(v, torch.Tensor) and not k=="avg_reward" else v)
                                for k,v in mini_batches[i].items()}
                    # forward + backward on this micro‑batch
                    loss = loss_fn(model=model, rollout_data=rollout) / num_reward_workers
                    total_loss_in_iter += loss.item()
                    # sum up gradients from two batches
                    loss.backward()
                
                wait = time.perf_counter() - t0 
                print(f"TIME to run 3 GRPO iterations on 2 mini batches{wait}", flush=True)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()

                avg_loss_this_iter = total_loss_in_iter
                print(f"Trainer (Rank {rank}): Epoch {epoch+1}, Step {step+1}, GRPO Iter {grpo_iter+1}/{config.batch_updates}, Loss: {avg_loss_this_iter:.4f}", flush=True)
                wandb.log({
                    "loss": avg_loss_this_iter,
                    "step": step,
                    "grpo_iter": grpo_iter + 1,
                    "epoch": epoch + 1,
                })
            
            wandb.log({"average_reward": avg_reward, "step": step, "epoch": epoch + 1})

            t0 = time.perf_counter()
            for p in model.parameters():
                dist.broadcast(p.data, src=rank)
            
            wait = time.perf_counter() - t0 
            print(f"TIME to sync parameters across devices {wait}", flush=True)

            step += 1

            del mini_batches
            torch.cuda.empty_cache()


    if rank == num_reward_workers:
        wandb.finish()
    return




def main(
    rank: int, world_size: int, queue, config: TrainConfig):
    print(f"main invoked as rank={rank}, world_size={world_size}", flush=True)
    os.environ["RANK"]= str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"]    = str(world_size)

    setup(world_size)
    torch.cuda.set_device(rank)

    attn_implementation = 'flash_attention_2' if torch.cuda.is_available() else None
    print(f"Rank {rank}: Initializing model", flush=True)
    model = Cadrille.from_pretrained(
        config.sft_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map=rank).train().to(rank)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",
                                              min_pixels=256 * 28 * 28,
                                              max_pixels=1280 * 28 * 28,
                                              padding_side="left",
                                              )

    eval_data_deepcad = RealDatasetMM(path=f'/home/jovyan/users/zhemchuzhnikov/tarasov/data/deepcad_test', file_name='test.pkl', n_points=256, size=1000)
    eval_data_fusion = RealDatasetMM(path=f'/home/jovyan/users/zhemchuzhnikov/tarasov/data/fusion360_test', file_name='test.pkl', n_points=256, size=1000)
    train_data = RealDatasetMM(path=f'/home/jovyan/users/zhemchuzhnikov/tarasov/data/deepcad_fusion_train', file_name=config.train_file, n_points=256, mode=config.train_mode, noise_scale_pc=0.01, size=config.train_size)
    print(f"Rank {rank}: Initializing datasets", flush=True)

    model = optimize_model_memory(model)

    #if rank == 1:
    #    model = DDP(model, device_ids=[rank], find_unused_parameters=True)


    print(f"\nRank {rank}: Starting RL fine-tuning using GRPO…")


    if rank < config.num_reward_workers:
        print(f"Rank {rank}: Starting reward inference worker", flush=True)
        reward_inference_worker(
            queue, model, processor, train_data, config, rank,
        )
    else:
        print(f"Rank {rank}: Starting trainer worker", flush=True)
        trainer_worker(
            queue, model, processor, config, rank,
        )
    cleanup()
    print("Training completed.")

@pyrallis.wrap()
def spawn_main(config: TrainConfig):

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "1245"
    
    # 1) parse config exactly once
    world_size = config.num_reward_workers + 1
    spawn_ctx = mp.get_context("spawn")
    queue = spawn_ctx.Queue(maxsize=2*config.num_reward_workers )
    mp.spawn(
        fn=main,
        nprocs=world_size,
        args=(world_size, queue, config,),
        join=True,
    )

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    spawn_main()
