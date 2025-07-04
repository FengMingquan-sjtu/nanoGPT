#!/bin/bash
# This script evaluates the perplexity of a model on the GSM8K dataset using the eval_ppl.py script.

model_path="out/cont-gpt2-1.5B-owm-15B/2025-07-02_21-50-52/"
wandb_id="ee485d6a"
gpu_id=0
checkpoints=(
    2000
    10000
    12000
    14000
    16000
    18000
    20000
    22000
    28000
    30000
)
for i in "${!checkpoints[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu_id nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python eval_ppl.py --model_path $model_path --wandb_id $wandb_id --ckpt_step ${checkpoints[$i]}
done
echo "All jobs completed."