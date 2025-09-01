import sys
import os
from dataclasses import asdict, dataclass
from functools import partial
from queue import Empty, Full

import time

os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["MKL_NUM_THREADS"]       = "1"

os.environ["NCCL_CUMEM_ENABLE"]     = "0"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

os.environ["TORCHELASTIC_ERROR_FILE"]   = "./error.json"
from torch.distributed.elastic.multiprocessing.errors import record
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import pyrallis
from comet_ml import ExperimentConfig, start
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp

from utils_cadrille import init_pool, close_pool

from grpo_mm import generate_rollout_data, grpo_loss, gspo_loss
from train_cadrille_grpo_base import TrainConfig, collate_img_pc_v1, get_reward_function, optimize_model_memory, setup, cleanup

from cad_recode_model_mm import Cadrille

from transformers import AutoProcessor
from dataset_utils import RealDatasetMM


def sync_params(model):
    # stable order, skip point_encoder
    return [p for n, p in sorted(model.named_parameters(), key=lambda x: x[0])
            if not n.startswith("point_encoder.")]

def alive(done_flags):
    return [i for i, d in enumerate(done_flags) if not d]

SEED = 16

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    DONE: str = "done"
    RANK: str = "rank"
    END_EPOCH: str = "end_epoch"
    IDX_TENSOR: str = "idx_tensor"


def reward_inference_worker(queue, model, processor, train_data, config, rank):
    model.eval()
    #model.gradient_checkpointing_disable()
    #model.config.use_cache = False

    print("Initializing Multiprocesssing pool")
    pool = init_pool(config.pool_size)

    torch.cuda.set_device(rank)

    sampler = DistributedSampler(train_data, num_replicas=config.num_reward_workers, rank=rank, shuffle=False)

    reward_function = get_reward_function(config.failure_reward, iou_coef=config.iou_coef, cd_coef=config.cd_coef, auc_coef=config.auc_coef,)

    step = 0

    #start_batch = 80
    #ctx = mp.get_context('spawn')
    dataloader = DataLoader(train_data, batch_size=config.batch_size // config.num_reward_workers, collate_fn=partial(collate_img_pc_v1, processor=processor, n_points=256), sampler=sampler,
                                num_workers=config.dataloader_workers)

    print(f"Dataloader len : {len(dataloader)}")

    print("Synchronizing initial parameters")
    sync_list = sync_params(model)

    with torch.no_grad():
        ref   = torch.nn.utils.parameters_to_vector(sync_list).detach()

    flat = torch.empty_like(ref, device=ref.device) 


    for epoch in range(config.train_epochs):

        print(f"Generator (Rank {rank}): Starting epoch {epoch + 1}/{config.train_epochs}.")
        sampler.set_epoch(epoch)

        #for i, batch in enumerate(islice(dataloader, start_batch, None), start=start_batch):
        for i, batch in enumerate(dataloader):
            # synchronize the model parameters from Trainer GPU 
            print(f"Generator (Rank {rank}): Synchronizing model parameters.")
            t0 = time.perf_counter()
            if step > -1 :
                dist.recv(flat, src=config.num_reward_workers)
                with torch.no_grad():
                    torch.nn.utils.vector_to_parameters(flat, sync_list)
            flat = flat.clone().contiguous()
            p_wait = time.perf_counter() - t0 
            print(f"TIME to receive parameters from Trainer : {p_wait} to rank {rank}")


            print(f"Generator (Rank {rank}) Generating rollouts for batch {step + 1}/{len(dataloader)}")
            torch.cuda.empty_cache()

            rollout, avg_reward = generate_rollout_data(
                model,
                reward_function,
                processor,
                batch,
                config.num_generations,
                config.max_completion_length,
                top_samples=config.top_samples,
                gpg=config.use_gpg,
                buffer = None,
                temperature=config.temperature,
                do_sample=config.do_sample,
                top_p=config.top_p,
                )
            

            t0 = time.perf_counter()

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
                IPCKeys.IDX_TENSOR,

            ]:
                if key in rollout:
                    if isinstance(rollout[key], torch.Tensor):
                        payload[key] = rollout[key].detach().cpu().clone()
                    else:
                        payload[key] = rollout[key]
            payload[IPCKeys.AVG_REWARD] = avg_reward
            payload[IPCKeys.RANK] = rank
            if epoch == (config.train_epochs - 1) and i == (len(dataloader) - 1):
                print(f"Sending end of training signal by Generator {rank}")
                payload[IPCKeys.DONE] = True
            
            t_create_payload = time.perf_counter() - t0
            print(f"[Rank {rank}] time to put rollout in shared memory : {t_create_payload})")

            t0 = time.perf_counter()
            try:
                queue.put(payload, timeout=320)
            except Full:
                print(f"[Rank {rank}] Queue is actually FULL (size: {queue.qsize()})")
                continue
            
            step+=1
            t_send_payload = time.perf_counter() - t0
            print(f"[Rank {rank}] time to share Payload tensor to Trainer : {t_send_payload})")


        # Signal to trainer that the epoch is finished
        queue.put({IPCKeys.END_EPOCH: True, IPCKeys.RANK: rank})
    print(f"Generator (Rank {rank}): All epochs complete.")

    

def trainer_worker(queue, model, processor, config, rank):

    """GPU 1: compute loss, update model, evaluate."""
    print(f"Starting Trainer (Rank {rank}) ")
    set_seed(SEED)
    torch.cuda.empty_cache()
    torch.cuda.set_device(rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


    reward_function = get_reward_function(config.failure_reward)
    rl_loss = grpo_loss if not config.use_gspo else gspo_loss
    loss_fn = partial(rl_loss, processor=processor,epsilon_high=config.epsilon_high, epsilon_low=config.epsilon_low, reward_function=reward_function)

    num_reward_workers = config.num_reward_workers
    done = [False] * config.num_reward_workers

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

    sync_list = sync_params(model)
    flat = torch.nn.utils.parameters_to_vector(sync_list).detach().clone().contiguous()

    for _ in range(2):
        for dst in range(config.num_reward_workers):
            dist.send(flat, dst=dst)

    skip_iters = 2
    skip_counter = 0

    for epoch in range(config.train_epochs):

        print(f"Trainer (Rank {rank}): Starting epoch {epoch + 1}/{config.train_epochs}.")
        end_signals_received = 0
        
        # Process training steps until all workers finish the epoch
        while True:
            # Collect mini-batches from all currently alive workers
            #print(f"Trainer torch.cuda.memory_summary() : {torch.cuda.memory_summary(abbreviated=False)}")

            mini_batches = []
            avg_rewards = []
            alive_workers = alive(done)

            nb_waits = 0

            if not alive_workers:
                print(f"Trainer (Rank {rank}): No alive workers remaining.")
                break

            if skip_counter > 0:
                print(f"Trainer : Skipping parameter broadcast to Reward workers skip iteration {skip_counter} iterations remaining")
                flat = torch.nn.utils.parameters_to_vector(sync_list).detach().clone().contiguous()
                for dst in alive_workers: 
                    dist.send(flat, dst)
                skip_counter -= 1
                step+=1
                continue


            #print(f"Trainer (Rank {rank}): Collecting data from {len(alive_workers)} alive workers")
            workers_responded = set()
            while len(mini_batches) + end_signals_received < len(alive_workers):
                print(f"len(mini_batches) {len(mini_batches)}, end_signals_received {end_signals_received}, len(alive_workers, {len(alive_workers)}")
                t0 = time.perf_counter()
                try:
                    item = queue.get(timeout=300)
                except Empty:
                    print(f"Empty queue, retrying... (waiting for {len(alive_workers) - len(workers_responded)} more responses)")
                    print(f"Workers responded so far: {workers_responded}")
                    print(f"Alive workers: {alive_workers}")
                    # Check if we've been waiting too long - might indicate end of epoch
                    # Instead of infinite waiting, let's check if some workers sent end-of-epoch signals
                    if end_signals_received > 0:
                        print(f"Some workers ({end_signals_received}) already sent end-of-epoch signals, breaking")
                        break
                    raise TimeoutError

                q_wait = time.perf_counter() - t0 
                print(f"TIME to get sample from queue {q_wait}")

                worker_rank = item.get(IPCKeys.RANK)

                if IPCKeys.AVG_REWARD in item:
                    if IPCKeys.DONE in item:
                        done[worker_rank] = True
                        print(f"[Trainer {rank}][E{epoch}] DATA from r{worker_rank} (LAST)")
                    
                    avg_rewards.append(item[IPCKeys.AVG_REWARD])
                    mini_batches.append(item)
                    workers_responded.add(worker_rank)
                    print(f"[Trainer {rank}][E{epoch}] DATA from r{worker_rank}. Total mini-batches: {len(mini_batches)}")


                elif IPCKeys.END_EPOCH in item:
                    print(f"Trainer (Rank {rank}): Received end-of-epoch signal from worker {worker_rank}")
                    end_signals_received += 1
                    workers_responded.add(worker_rank) 

                elif IPCKeys.DONE in item:
                    done[worker_rank] = True
                    print(f"[Trainer {rank}][E{epoch}] DONE signal from r{worker_rank}")
                    end_signals_received += 1
                    workers_responded.add(worker_rank)
                
                del item

                if q_wait > 0.5:
                    nb_waits +=1

                if nb_waits == len(alive_workers):
                    skip_counter = skip_iters
                    break
            
            if end_signals_received >= len(alive_workers):
                print(f"Trainer (Rank {rank}): Received all {end_signals_received} end-of-epoch signals for epoch {epoch + 1}")
                break
            
            if skip_counter > 0:
                continue
            
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            if mini_batches:
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
                        print(f"mini batch {i}")
                        # move tensors to GPU
                        rollout = {k: (v.to(rank) if isinstance(v, torch.Tensor) and not k=="avg_reward" else v)
                                    for k,v in mini_batches[i].items()}
                        # forward + backward on this micro‑batch
                        loss = loss_fn(model=model, rollout_data=rollout) / len(mini_batches)
                        total_loss_in_iter += loss.item()
                        # sum up gradients from two batches
                        loss.backward()
                    
                    #print(f"TIME to run 1 GRPO iterations on {num_reward_workers} mini batches {wait}")

                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                    optimizer.step()

                    wait = time.perf_counter() - t0 

                    avg_loss += total_loss_in_iter
                    print(f"Trainer (Rank {rank}): Epoch {epoch+1}, Step {step+1}, GRPO Iter {grpo_iter+1}/{config.batch_updates}, Loss: {total_loss_in_iter}")
                    experiment.log_metrics({
                        "loss": total_loss_in_iter,
                        "grpo_iter": grpo_iter + 1,
                        "epoch": epoch + 1,
                        "grad_norm": norm.item(),
                        "time/iter_s": wait,
                        "trainer_step":step,
                    })

                experiment.log_metrics({"average_reward": avg_reward, "epoch": epoch + 1, 
                        "time/queue_s": q_wait, 
                        "mean_advantage_0": mini_batches[0][IPCKeys.ADV].mean().item(),
                        "loss_avg": avg_loss / config.batch_updates
                        }, step = step)


            t0 = time.perf_counter()
            flat = torch.nn.utils.parameters_to_vector(sync_list).detach().clone().contiguous()
            workers_with_data = {mb[IPCKeys.RANK] for mb in mini_batches}
            for dst in workers_with_data:
                if not done[dst]:
                    print(f"[Trainer {rank}][E{epoch}] sending params to {dst}")
                    dist.send(flat, dst=dst)

            wait = time.perf_counter() - t0 
            print(f"TIME to broadcast parameters across devices from Trainer : {wait}")

            step += 1

            if mini_batches:
                del mini_batches

            if config.save_mid_epoch and (step % ( 52320 // (config.batch_size * 2))) == 0 :
                model.save_pretrained(f"{config.save_path}/{config.name}_{epoch}_mid")
                processor.save_pretrained(f"{config.save_path}/{config.name}_{epoch}_mid")


        print("Emptying cuda cache and saving model")
        torch.cuda.empty_cache()
        model.save_pretrained(f"{config.save_path}/{config.name}_{epoch}")
        processor.save_pretrained(f"{config.save_path}/{config.name}_{epoch}")


    if rank == num_reward_workers:
        experiment.end()
    return




@record
def main(
    rank: int, world_size: int, queue, config: TrainConfig):
    print(f"main invoked as rank={rank}, world_size={world_size}")
    os.environ["RANK"]= str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"]    = str(world_size)

    torch.cuda.set_device(rank)
    setup(world_size)


    attn_implementation = 'flash_attention_2' if torch.cuda.is_available() else None
    print(f"Rank {rank}: Initializing model")
    model = Cadrille.from_pretrained(
        config.sft_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map=rank).train().to(device = torch.device(f"cuda:{rank}"))

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


    # freeze embeddings
    for p in model.get_input_embeddings().parameters():
        p.requires_grad = False

    #if rank == 1:
    #    model = DDP(model, device_ids=[rank], find_unused_parameters=True)


    print(f"\nRank {rank}: Starting RL fine-tuning using GRPO with PID {os.getpid()}")

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
    close_pool()
    dist.barrier()
    cleanup()
    print("Training completed.")

@pyrallis.wrap()
def spawn_main(config: TrainConfig):

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    #os.environ["MASTER_PORT"] = "1240"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    world_size = config.num_reward_workers + 1
    mp.set_start_method("spawn")

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
