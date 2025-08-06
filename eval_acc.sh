#!/bin/bash
# This script evaluates the accuracy of a model on the GSM8K dataset using the eval_acc.py script.
# Usage:
#   cd /cpfs/user/fengmingquan/nanoGPT
#   nohup bash eval_acc.sh > log/eval_acc_0.out 2>&1 &


model_path="out/cont-tinyllama-1.1B-tinypajama-0.6rho"
model_name="auto"  # Specify the model name
wandb_id="auto"  # Set to "auto" to automatically find the wandb ID from the log file
gpu_id=7
batch_size=30  # Adjust batch size as needed
block_size=2048  # Adjust block size as needed
n_shot_prompt=5  # Number of n-shot examples to include in the prompt
max_batches=100000000  # Maximum number of batches to process
device="cuda"


checkpoints=(
    0 1000 2000 4000 6000 8000 10000
    12000 14000 16000 18000 20000
#    22000 24000 26000 28000 30000
)
datasets=(
    "boolq"
)
# arc_challenge arc_easy hellaswag boolq piqa winogrande openbookqa  
for i in "${!checkpoints[@]}"; do
    for j in "${!datasets[@]}"; do
        dataset_name=${datasets[$j]}
        echo "Evaluating checkpoint ${checkpoints[$i]} on dataset $dataset_name"
        CUDA_VISIBLE_DEVICES=$gpu_id /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python eval_acc.py --model_path $model_path --wandb_id $wandb_id --ckpt_step ${checkpoints[$i]} --model_name $model_name --batch_size $batch_size --block_size $block_size --dataset_name $dataset_name --n_shot_prompt $n_shot_prompt --max_batches $max_batches --num_proc_load_dataset 8 --device $device
    done
done
echo "All jobs completed."