import wandb
import os

from multiprocessing import Process
from functools import partial

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'accuracy', 'goal': 'maximize'},
    'parameters': {
        'lr': {'values': [1e-4, 5e-4, 1e-5, 5e-5, 1e-6, 5e-6, 1e-7]},
        'sow_lr': {'values': [1e-4, 5e-4, 1e-5, 5e-5, 1e-6, 5e-6, 1e-7]},
        'rank': {'values': [1, 2, 5, 10, 20]},
        'sow_accumulation': {'values': [250, 500, 1000, 1500]},
    }
}

def train(gpu_id):
    with wandb.init(mode="disabled") as run:
        config = run.config
        run.name = (
            "sow_lr_" + f"{config.lr:.2e}" 
            + "_slr_" + f"{config.lr:.2e}"
            + "_rank_" + str(config.rank)
            + "_acc_" + str(config.sow_accumulation)
        )    

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.system(f"CUDA_VISIBLE_DEVICES={gpu_id} python run_glue.py \
        --model_name_or_path roberta-base \
        --task_name sst2 \
        --max_length 512 \
        --seed=42 \
        --per_device_train_batch_size 16 \
        --learning_rate {config.lr} \
        --sow_lr {config.sow_lr} \
        --num_train_epochs 30 \
        --architecture sow \
        --rank {config.rank} \
        --init_method normal_QR \
        --accumulation_steps {config.sow_accumulation} \
        --output_dir results/ft/roberta_base/sst2")

sweep_id = wandb.sweep(sweep_config, project="glue_sst2")
gpu_pools = ["2", "3", "4", "5"]

def run(gpu_id):
    wandb.agent(sweep_id, function=partial(train, gpu_id=gpu_id))

processes = []
for gpu_id in gpu_pools:
    p = Process(target=run, args=(gpu_id,))
    p.start()

for p in processes:
    p.join()

    

