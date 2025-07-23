import os
from dataclasses import asdict, dataclass
from datetime import timedelta
from functools import partial
from multiprocessing import Manager
from queue import Empty
from itertools import islice

import time

import pyrallis
from comet_ml import ExperimentConfig, start
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp

from utils import init_pool
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["OPENBLAS_NUM_THREADS"]  = "1"
os.environ["MKL_NUM_THREADS"]       = "1"

from grpo_mm import generate_rollout_data, grpo_loss, compute_log_probs
from train_cadrille_grpo import TrainConfig, collate_img_pc_v1, get_reward_function, optimize_model_memory, setup, cleanup

from cad_recode_model_mm import Cadrille

from transformers import AutoProcessor
from dataset_utils import RealDatasetMM


def init_flag(device):
    return torch.zeros(1, dtype=torch.long, device=device)

def push(flag, param_ver, src):
    flag.fill_(param_ver)                       # write new version
    dist.broadcast(flag, src)

def pull(flag, src):
    dist.broadcast(flag, src)              # blocking receive
    return int(flag.item())


def push_state(ddp_model, version):
    if dist.get_rank() == 0:                 # the “authority” trainer
        obj = {"ver": version,
               "state": ddp_model.module.state_dict()}
    else:
        obj = None
    dist.broadcast_object_list([obj], src=0, group=reward_pg)
    return obj

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
    print("Initializing Multiprocesssing pool")
    init_pool(6)

    torch.cuda.set_device(rank)

    sampler = DistributedSampler(train_data, num_replicas=config.num_reward_workers, rank=rank, shuffle=True)

    reward_fn = get_reward_function(config.failure_reward)

    step = 0

    #start_batch = 80
    dataloader = DataLoader(train_data, batch_size=config.batch_size // config.num_reward_workers, collate_fn=partial(collate_img_pc_v1, processor=processor, n_points=256), sampler=sampler,
                                num_workers=4)

    print(f"Datalaoder len : {len(dataloader)}")
    print(f"Setting up handshake flag")

    flag = init_flag(rank)
    last_param_ver = 0

    for epoch in range(config.train_epochs):

        print(f"Generator (Rank {rank}): Starting epoch {epoch + 1}/{config.train_epochs}.")
        sampler.set_epoch(epoch)

        #for i, batch in enumerate(islice(dataloader, start_batch, None), start=start_batch):
        for i, batch in enumerate(dataloader):
            # synchronize the model parameters from Trainer GPU 
            print(f"Generator (Rank {rank}): Synchronizing model parameters.")
            if step > -1 :
                ver = pull(flag, src=config.num_reward_workers)
                if ver != last_param_ver:
                    print(f"Generating samples from {ver} parameters")
                    for param in model.parameters():
                        dist.broadcast(param.data, src=config.num_reward_workers)
                    last_param_ver = ver
            print(f"Generator (Rank {rank}) batch {i}")

            print(f"Generator (Rank {rank}) Generating rollouts for batch {step + 1}/{len(dataloader)}")
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

    """GPU 1: compute loss, update model, evaluate."""
    print(f"Starting Trainer (Rank {rank}) ")
    torch.cuda.set_device(rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


    reward_function = get_reward_function(config.failure_reward)

    loss_fn = partial(grpo_loss, processor=processor,epsilon_high=config.epsilon_high, epsilon_low=config.epsilon_low, reward_function=reward_function)

    num_reward_workers = config.num_reward_workers

    experiment_config = ExperimentConfig(
        auto_output_logging= "False",
        auto_param_logging=True,
        auto_histogram_activation_logging=True
    )

    experiment = start(
        api_key="CfQGtyWGF13CZEsUvXBeuPaSf",
        project_name="cad",
        workspace="marinabar",
        experiment_config=experiment_config
    )

    experiment.set_name(config.name)

    params = {k: getattr(config, k) for k in config.__annotations__}
    experiment.log_parameters(params)

    step = 0
    optimizer.zero_grad()

    #push(flag, step=0, src=rank)

    print(f"Setting up handshake flag")
    flag = init_flag(rank)
    global_ver = 0

    push(flag, param_ver=0, src=rank)
    push(flag, param_ver=0, src=rank)

    for epoch in range(config.train_epochs):

        print(f"Trainer (Rank {rank}): Starting epoch {epoch + 1}/{config.train_epochs}.")
        end_signals = 0

        while end_signals < num_reward_workers:
            mini_batches = []
            avg_rewards = []
            while len(mini_batches) != num_reward_workers:
                t0 = time.perf_counter()
                try:
                    item = queue.get(timeout=250)
                except Empty:
                    print("Empty queue")
                    for p in model.parameters():
                        dist.broadcast(p.data, src=rank)
                    break

                q_wait = time.perf_counter() - t0 
                print(f"TIME to get sample from queue {q_wait}")
                if item is None:
                    print(f"Trainer (Rank {rank}): Received end-of-epoch signal from one worker.")
                    end_signals += 1
                    continue

                avg_rewards.append(item["avg_reward"])

                mini_batches.append(item)
            
            if not mini_batches:
                print(f"Trainer (Rank {rank}): Received {num_reward_workers} end-of-epoch signals.")
                for p in model.parameters():
                    dist.broadcast(p.data, src=rank)
                continue

            # compute the average reward across that concatenated batch
            avg_reward = sum(avg_rewards) / len(avg_rewards)

            avg_loss = 0
            # parameter updates following the direction of the loss
            for grpo_iter in range(config.batch_updates):
                t0 = time.perf_counter()
                optimizer.zero_grad()
                total_loss_in_iter = 0

                #gradient accumulation
                for i in range(len(mini_batches)):
                    # move tensors to GPU
                    rollout = {k: (v.to(rank) if isinstance(v, torch.Tensor) and not k=="avg_reward" else v)
                                for k,v in mini_batches[i].items()}
                    # forward + backward on this micro‑batch
                    loss = loss_fn(model=model, rollout_data=rollout) / len(mini_batches)
                    total_loss_in_iter += loss.item()
                    # sum up gradients from two batches
                    loss.backward()
                
                wait = time.perf_counter() - t0 
                print(f"TIME to run 1 GRPO iterations on {num_reward_workers} mini batches {wait}")

                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()

                avg_loss += total_loss_in_iter
                print(f"Trainer (Rank {rank}): Epoch {epoch+1}, Step {step+1}, GRPO Iter {grpo_iter+1}/{config.batch_updates}, Loss: {total_loss_in_iter:.4f}")
                experiment.log_metrics({
                    "loss": total_loss_in_iter,
                    "step": step +1,
                    "grpo_iter": grpo_iter + 1,
                    "epoch": epoch + 1,
                    "grad_norm": norm.item(),
                    "time/iter_s": wait,
                })
            
            experiment.log_metrics({"average_reward": avg_reward, "step": step+1, "epoch": epoch + 1, 
                    "time/queue_s": q_wait, 
                    "mean_advantage_0": mini_batches[0][IPCKeys.ADV].mean().item(),
                    "loss_avg": avg_loss / config.batch_updates
                    })


            t0 = time.perf_counter()
            #for p in model.parameters():
                #dist.broadcast(p.data, src=rank)
            global_ver += 1
            push(flag, global_ver, src=rank)
            for p in model.parameters():
                dist.broadcast(p.data, src=rank)
            wait = time.perf_counter() - t0 
            print(f"TIME to broadcast parameters across devices from Trainer : {wait}")

            step += 1

            del mini_batches
            torch.cuda.empty_cache()


    if rank == num_reward_workers:
        experiment.end()
    return




def main(
    rank: int, world_size: int, queue, config: TrainConfig):
    print(f"main invoked as rank={rank}, world_size={world_size}")
    os.environ["RANK"]= str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"]    = str(world_size)

    setup(world_size)
    torch.cuda.set_device(rank)

    attn_implementation = 'flash_attention_2' if torch.cuda.is_available() else None
    print(f"Rank {rank}: Initializing model")
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
    print(f"Rank {rank}: Initializing datasets")

    model = optimize_model_memory(model)

    #if rank == 1:
    #    model = DDP(model, device_ids=[rank], find_unused_parameters=True)


    print(f"\nRank {rank}: Starting RL fine-tuning using GRPO…")


    if rank < config.num_reward_workers:
        print(f"Rank {rank}: Starting reward inference worker")
        reward_inference_worker(
            queue, model, processor, train_data, config, rank,
        )
    else:
        print(f"Rank {rank}: Starting trainer worker")
        trainer_worker(
            queue, model, processor, config, rank,
        )
    cleanup()
    print("Training completed.")

@pyrallis.wrap()
def spawn_main(config: TrainConfig):

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "1240"
    os.environ.pop("COMET_AUTO_OUTPUT_LOGGING", None)
    
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
    spawn_main()
