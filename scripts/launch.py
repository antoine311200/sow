import wandb
import os

# wandb.login()


# os.system("torchrun --standalone --nproc_per_node 4 ./simple_train.py \
#     --model_config ./configs/llama_350m.json \
#     --lr 0.001 \
#     --batch_size 64 \
#     --total_batch_size 128 \
#     --num_training_steps 10000 \
#     --warmup_steps 250 \
#     --weight_decay 0 \
#     --dtype bfloat16 \
#     --optimizer adam \
#     --architecture slinear \
#     --rank 10 \
#     --n_iter 5 \
#     --monitor_memory false")
    # --single_gpu \


# Tensor Train Sum Linear layer
# os.system("torchrun --standalone --nproc_per_node 1 ./simple_train.py \
#     --model_config ./configs/llama_60m.json \
#     --lr 0.0015 \
#     --batch_size 128 \
#     --total_batch_size 256 \
#     --num_training_steps 10000 \
#     --warmup_steps 1000 \
#     --weight_decay 0 \
#     --dtype bfloat16 \
#     --optimizer adam \
#     --architecture slinear \
#     --rank 10 \
#     --inner_rank 4 \
#     --order 3 \
#     --n_iter 5 \
#     --single_gpu \
    # --monitor_memory true")


# os.environ["CUDA_VISIBLE_DEVICES"] = "8"

# os.system("torchrun --standalone --nproc_per_node 1 ./simple_train.py \
#     --model_config ./configs/llama_1b.json \
#     --lr 0.005 \
#     --batch_size 16 \
#     --total_batch_size 512 \
#     --num_training_steps 150000 \
#     --warmup_steps 1500 \
#     --weight_decay 0 \
#     --grad_clipping 1.0 \
#     --dtype bfloat16 \
#     --optimizer galore_adamw \
#     --scheduler cosine \
#     --architecture galore \
#     --galore_scale 0.25 \
#     --galore_rank 512 \
#     --update_proj_gap 500 \
#     --sow_accumulation 2000 \
#     --sow_lr 0.001 \
#     --lr_decay 0.85 \
#     --rank 20 \
#     --n_iter 10 \
#     --single_gpu \
#     --monitor_memory true")

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# os.system("torchrun --standalone --nproc_per_node 1 ./simple_train.py \
#     --model_config ./configs/llama_1b.json \
#     --lr 0.0015 \
#     --batch_size 64 \
#     --total_batch_size 128 \
#     --num_training_steps 300000 \
#     --warmup_steps 0.1 \
#     --weight_decay 0 \
#     --dtype bfloat16 \
#     --scheduler cosine \
#     --optimizer adam \
#     --eval_every 40000 \
#     --save_every 39999 \
#     --grad_clipping 1.0 \
#     \
#     --architecture linear \
#     --sow_accumulation 10000 \
#     --sow_lr 0.00085 \
#     --lr_decay 0.85 \
#     --rank 20 \
#     --n_iter 10 \
#     \
#     --single_gpu \
#     --monitor_memory true")
    # --activation_checkpointing \

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"

os.system("torchrun --standalone --nproc_per_node 4 ./simple_train.py \
    --model_config ./configs/llama_350m.json \
    --lr 0.001 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 50000 \
    --warmup_steps 0.05 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --scheduler cosine \
    --save_every 100000 \
    --single_gpu \
    --monitor_memory true \
    --min_lr_ratio 0.03 \
    --max_length 256 \
    --optimizer galore_adamw \
    \
    --architecture linear \
    --sow_accumulation 500 \
    --sow_lr 0.00175 \
    --init_method kaiming_normal \
    --sow_scale 1.0 \
    --rank 100 \
    --n_iter 1 \
    \
    ")
    # --lr_decay 0.85 \
    # --reset_scheduler \
    # --accumulate_after_warmup \

# Evaluation
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.system("torchrun --standalone --nproc_per_node 1 ./simple_train.py \
#     --eval \
#     --model_config ./configs/llama_350m.json \
#     --continue_from ./checkpoints/llama_350m-2024-10-06-22-51-56/model_60000\
#     --batch_size 96 \
#     --total_batch_size 768 \
#     --dtype bfloat16 \
#     --single_gpu \
#     --seed 3112 \
#     \
#     --architecture sow \
#     --rank 200 \
#     --n_iter 1 \
#     \
# ")