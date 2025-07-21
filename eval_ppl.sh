#!/bin/bash
# This script evaluates the perplexity of a model on the GSM8K dataset using the eval_ppl.py script.
# Usage:
#   nohup bash eval_ppl.sh > log/eval_ppl_0.out 2>&1 &
model_path="out/cont-qwen2-1.5B-owm-15B/2025-07-21_09-58-25"
model_name="Qwen/Qwen2-1.5B"  # Specify the model name
wandb_id="y883tj0q"
gpu_id=0
batch_size=30  # Adjust batch size as needed

checkpoints=(
    0 2000 4000 6000 8000
)
#    10000 12000 14000 16000 18000
#    20000 22000 24000 26000 28000
#    30000)
for i in "${!checkpoints[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu_id /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python eval_ppl.py --model_path $model_path --wandb_id $wandb_id --ckpt_step ${checkpoints[$i]} --model_name $model_name --batch_size $batch_size
done
echo "All jobs completed."