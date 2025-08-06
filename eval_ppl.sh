#!/bin/bash
# This script evaluates the perplexity of a model on the GSM8K dataset using the eval_ppl.py script.
# Usage:
#   cd /cpfs/user/fengmingquan/nanoGPT
#   nohup bash eval_ppl.sh > log/eval_ppl_0.out 2>&1 &

# Math model list:
#out/cont-kinetgpt2-0.1B-owm-15B
#out/cont-kinetgpt2-0.3B-owm-15B
#out/cont-kinetgpt2-0.7B-owm-15B
#out/cont-kinetgpt2-1.5B-owm-15B
#out/cont-gpt2-124M-owm-7.5B-1.0rho
#out/cont-gpt2-0.3B-owm-15B
#out/cont-gpt2-0.7B-owm-15B
#out/cont-gpt2-1.5B-owm-15B

# Gneral model list:
#out/cont-gpt2-0.1B-owt-9B
#out/cont-gpt2-0.3B-owt-9B
#out/cont-gpt2-0.7B-owt-9B
#out/cont-gpt2-1.5B-owt-9B
#out/cont-kinetgpt2-0.1B-owt-15B
#out/cont-kinetgpt2-0.3B-owt-15B
#out/cont-kinetgpt2-0.7B-owt-15B
#out/cont-kinetgpt2-1.5B-owt-15B

model_path="out/cont-tinyllama-1.1B-tinypajama-0.6rho"
model_name="auto"  # Specify the model name
wandb_id="auto"  # Set to "auto" to automatically find the wandb ID from the log file
gpu_id=1
batch_size=15  # Adjust batch size as needed
block_size=2048  # Adjust block size as needed
n_shot_prompt=5  # Number of n-shot examples to include in the prompt
n_processes=8


checkpoints=(
    0 1000 2000 4000 6000 8000 10000
    12000 14000 16000 18000 20000
#    22000 24000 26000 28000 30000
)
datasets=(
    "hellaswag" "arc_c" "arc_e" "math" "gsm8k" "ocw" "mathqa" "bbh" "gpqa"
)
# math gsm8k ocw mathqa    bbh gpqa
# arc_c arc_e hellaswag
for i in "${!checkpoints[@]}"; do
    for j in "${!datasets[@]}"; do
        dataset_name=${datasets[$j]}
        echo "Evaluating checkpoint ${checkpoints[$i]} on dataset $dataset_name"
        CUDA_VISIBLE_DEVICES=$gpu_id /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python eval_ppl.py --model_path $model_path --wandb_id $wandb_id --ckpt_step ${checkpoints[$i]} --model_name $model_name --batch_size $batch_size --block_size $block_size --dataset_name $dataset_name --n_shot_prompt $n_shot_prompt --num_proc_load_dataset $n_processes
    done
done
echo "All jobs completed."