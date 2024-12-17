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

os.environ["CUDA_VISIBLE_DEVICES"] ="3,4,5"

    # --continue_from ./checkpoints/llama_60m-2024-11-06-11-33-32/model_5000 \
command = (
    f"torchrun --standalone --nproc_per_node 3 ./simple_train.py "
    f"--model_config ./configs/llama_1b.json "
    f"--lr 0.0001 "
    # f"--activation_checkpointing "
    f"--batch_size 48 "
    f"--total_batch_size 192 "
    f"--num_training_steps 200000 "
    f"--warmup_steps 0.05 "
    f"--weight_decay 0 "
    f"--dtype bfloat16 "
    f"--scheduler cosine "
    f"--save_every 2500000 "
    f"--eval_every 3999 "
    # f"--single_gpu "
    f"--monitor_memory true "
    f"--min_lr_ratio 0.03 "
    f"--max_length 256 "
    f"--optimizer adamw "

    f"--architecture sow "
    f"--sow_accumulation 5000 "
    f"--sow_lr 0.00025 "
    f"--rank 200 "
    # f"--wandb_off"
)
os.system(command)
    # --eval \
    # --wandb_off \
    # --activation_checkpointing \
    # --lr_decay 0.85 \
    # --reset_scheduler \
    # --accumulate_after_warmup \

# Evaluation
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.system("torchrun --standalone --nproc_per_node 1 ./simple_train.py \
#     --eval \
#     --model_config ./configs/llama_60m.json \
#     --continue_from ./checkpoints/llama_60m-2024-10-31-11-37-40/model_260\
#     --batch_size 96 \
#     --total_batch_size 768 \
#     --dtype bfloat16 \
#     --single_gpu \
#     --seed 3112 \
#     \
#     --architecture sow \
#     --rank 100 \
#     --n_iter 1 \
#     \
# ")
