#!/bin/bash
# This script evaluates the perplexity of a model on the GSM8K dataset using the eval_ppl.py script.
# Usage:
#   cd /cpfs/user/fengmingquan/nanoGPT
#   nohup bash eval_ppl.sh > log/eval_ppl_5.out 2>&1 &
model_path="out/cont-tinyllama-1.1B-owm-7.5B-0.4rho/2025-07-24_13-01-05"
model_name="auto"  # Specify the model name
wandb_id="auto"  # Set to "auto" to automatically find the wandb ID from the log file
gpu_id=6
batch_size=16  # Adjust batch size as needed
block_size=2048  # Adjust block size as needed
dataset_name="math"  # Specify the dataset name, e.g., gsm8k or math

checkpoints=(
    2000
)
#    10000 12000 14000 16000 18000
#    20000 22000 24000 26000 28000
#    30000)
for i in "${!checkpoints[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu_id /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python eval_ppl.py --model_path $model_path --wandb_id $wandb_id --ckpt_step ${checkpoints[$i]} --model_name $model_name --batch_size $batch_size --block_size $block_size --dataset_name $dataset_name 
done
echo "All jobs completed."