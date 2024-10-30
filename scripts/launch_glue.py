import wandb
import os

# wandb.login()

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

os.system("python run_glue.py \
    --model_name_or_path roberta-base \
    --task_name mrpc \
    --max_length 512 \
    --seed=1234 \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-5 \
    --sow_lr 2e-4 \
    --scale 8 \
    --num_train_epochs 30 \
    --architecture sow \
    --n_iter 1 \
    --rank 20 \
    --accumulation_steps 800 \
    --output_dir results/ft/roberta_base/mrpc")