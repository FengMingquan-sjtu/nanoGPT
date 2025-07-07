#!/bin/bash
# This script evaluates the perplexity of a model on the GSM8K dataset using the eval_ppl.py script.
# Usage:
#   nohup bash eval_ppl.sh > log/eval_ppl_0.out 2>&1 &
model_path="out/cont-gpt2-0.7B-owm-7.5B-0.8rho/2025-07-04_20-00-17"
model_arch="gpt2-large"  # Specify the model architecture, must be one of ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
wandb_id="d5dck7ho"
gpu_id=4

checkpoints=(
    0 2000 4000 6000 8000
    10000 12000 14000 16000 18000
    20000 22000 24000 26000 28000
    30000
)
for i in "${!checkpoints[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu_id /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python eval_ppl.py --model_path $model_path --wandb_id $wandb_id --ckpt_step ${checkpoints[$i]} --model_arch $model_arch
done
echo "All jobs completed."