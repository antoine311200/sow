import wandb
import os

# wandb.login()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

os.system("python run_glue.py \
    --model_name_or_path roberta-base \
    --task_name mrpc \
    --max_length 512 \
    --seed=1234 \
    --per_device_train_batch_size 16 \
    --learning_rate 3e-5 \
    --num_train_epochs 30 \
    --enable_sow \
    --n_iter 10 \
    --rank 10 \
    --accumulation_steps 500 \
    --output_dir results/ft/roberta_base/mrpc")