import os
from itertools import product
from multiprocessing import Process, Queue
import time

def run_glue(rank, freq, lr, sow_lr, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    command = (
        f"torchrun --standalone --nproc_per_node 1 ./simple_train.py "
        f"--model_config ./configs/llama_60m.json "
        f"--lr {lr} "
        f"--batch_size 128 "
        f"--total_batch_size 256 "
        f"--num_training_steps 25000 "
        f"--warmup_steps 0.05 "
        f"--weight_decay 0 "
        f"--dtype bfloat16 "
        f"--scheduler cosine "
        f"--save_every 2500000 "
        f"--eval_every 999 "
        f"--single_gpu "
        f"--monitor_memory true "
        f"--min_lr_ratio 0.03 "
        f"--max_length 256 "
        f"--optimizer adamw "
        f"--architecture sow "
        f"--sow_accumulation {freq} "
        f"--sow_lr {sow_lr} "
        f"--init_method normal_QR "
        f"--rank {rank} "
    )
    os.system(command)

def worker(job_queue, gpu_id):
    while not job_queue.empty():
        try:
            rank, acc, lr, sow_lr = job_queue.get_nowait()  # Fetch a job if available
            print(f"Starting job: rank={rank}, acc={acc}, lr={lr}, sow_lr={sow_lr} on GPU {gpu_id}")
            run_glue(rank, acc, lr, sow_lr, gpu_id)
            print(f"Finished job: rank={rank}, acc={acc}, lr={lr}, sow_lr={sow_lr} on GPU {gpu_id}")
        except Exception as e:
            print(f"Error in worker on GPU {gpu_id}: {e}")
            break

if __name__ == "__main__":
    gpu_ids = [0, 1, 2, 3]
    
    ranks = [25, 50, 100, 200]
    accs = [500, 1000, 2000, 5000]
    lrs = [0.01, 0.005, 0.001]
    sow_lrs = [0.01, 0.005, 0.001]
    param_combinations = list(product(ranks, accs, lrs, sow_lrs))
    print(f"Launching sweep with {len(param_combinations)} runs")
    
    # Create a job queue
    job_queue = Queue()
    for params in param_combinations:
        job_queue.put(params)
    
    # Start a worker process for each GPU
    processes = []
    for gpu_id in gpu_ids:
        p = Process(target=worker, args=(job_queue, gpu_id))
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()