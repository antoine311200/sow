import wandb
import os

# wandb.login()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.system("torchrun --standalone --nproc_per_node 1 ./simple_train.py \
    --model_config ./configs/roberta.json \
    --lr 0.02 \
    --order 3 \
    --rank 3 \
    --batch_size 32 \
    --total_batch_size 64 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --single_gpu \
    --optimizer adamw")
