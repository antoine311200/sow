# Sum-of-Weights (SoW)

### Pretraining Llama models

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node 3 ./scripts/simple_train.py \
    --model_config ./scripts/configs/llama_60m.json \
    --batch_size 128 \
    --total_batch_size 256 \
    --num_training_steps 25000 \
    --lr 0.01 \
    --warmup_steps 0.05 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --scheduler cosine \
    --save_every 10000 \
    --eval_every 2000 \
    --min_lr_ratio 0.03 \
    --max_length 256 \
    --optimizer adamw \
    --architecture sow \
    --sow_accumulation 5000 \
    --sow_lr 0.001 \
    --rank 50 \
```

### Finetuning on GLUE

For Roberta models,
```bash
CUDA_VISIBLE_DEVICES=0 python ./scripts/run_glue.py \
    --model_name_or_path roberta-base \
    --output_dir './trained_models/bool' \
    --task_name google/boolq \
    --max_length 512 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 30 \
    --eval_every 4000 \
    --learning_rate 5e-5 \
    --architecture sow \
    --sow_lr 1.2e-4 \
    --rank 8 \
    --accumulation_steps 1000 \
    --mode keep \
```

For Llama models,
```bash
CUDA_VISIBLE_DEVICES=0 python ./scripts/run_glue.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir './trained_models/bool' \
    --eval_llama \
    --activation_checkpointing \
    --task_name google/boolq \
    --max_length 512 \
    --seed=4321 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 30 \
    --eval_every 4000 \
    --learning_rate 5e-5 \
    --architecture sow \
    --sow_lr 1.2e-4 \
    --rank 8 \
    --mode keep \
    --accumulation_steps 1000 \
```

### Finetuning on Commonsense

```bash
CUDA_VISIBLE_DEVICES=0 python ./scripts/finetune.py \
  --base_model /PATH/llama-2-7b \
  --data_path /PATH/dataset/ARC-Easy \
  --output_dir ./trained_models/llama2-sow-arce \
  --batch_size 16 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 2e-5 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name 'sow' \
  --sow_lr 1.2e-4 \
  --accumulation_steps 200 \
  --rank 8 \
  --wandb_project arce
```

Evalute the trained model

```bash
CUDA_VISIBLE_DEVICES=8 python ./scripts/commonsense_evaluate.py \
    --model LLaMA-7B \
    --dataset /PATH/dataset/ARC-Easy \
    --base_model ./trained_models/llama2-sow-arce \
    --adapter sow \
    --batch_size 16
```