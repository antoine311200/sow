# import wandb
# import os

# # wandb.login()

# os.environ["CUDA_VISIBLE_DEVICES"] = "9"

# from itertools import product

# for rank, acc in product([8], [10000]):
#     os.system(f"python run_glue.py \
#         --model_name_or_path roberta-base \
#         --task_name rte \
#         --max_length 512 \
#         --seed=1234 \
#         --per_device_train_batch_size 16 \
#         --learning_rate 1e-5 \
#         --sow_lr 1e-5 \
#         --num_train_epochs 30 \
#         --eval_every 4000 \
#         --architecture sow \
#         --rank {rank} \
#         --init_method normal_QR \
#         --mode keep \
#         --accumulation_steps {acc} \
#         --output_dir results/ft/roberta_base/rte")
import os
from itertools import product
from multiprocessing import Process, Queue
import time

def run_glue(rank, acc, lr, sow_lr, gpu_id):
    """
    Function to run the GLUE task with specified parameters on a given GPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    command = (
        f"python run_glue.py "
        f"--model_name_or_path roberta-base "
        # f"--model_name_or_path huggyllama/llama-7b "
        # f"--eval_llama "
        f"--task_name google/boolq "
        f"--max_length 512 "
        f"--seed=4321 "
        f"--per_device_train_batch_size 16 "
        f"--learning_rate {lr} "
        f"--sow_lr {sow_lr} "
        f"--num_train_epochs 30 "
        f"--eval_every 4000 "
        f"--architecture sow "
        f"--rank {rank} "
        f"--init_method normal_QR "
        f"--mode keep "
        f"--scale 1 "
        f"--accumulation_steps {acc} "
        f"--output_dir results/ft/roberta_base/google-boolq"
    )
    os.system(command)

def worker(job_queue, gpu_id):
    """
    Worker process that pulls jobs from the queue and runs them on the assigned GPU.
    """
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
    # Define GPU IDs
    gpu_ids = [2]#3, 6, 7, 8]
    
    # Define parameter combinations
    ranks = [20]
    accs = [4000]  # Example multiple values
    lrs = [5e-6]
    sow_lrs = [7.5e-5]
    param_combinations = list(product(ranks, accs, lrs, sow_lrs))
    
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