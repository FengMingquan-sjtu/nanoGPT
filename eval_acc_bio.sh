#!/bin/bash
# This script evaluates the accuracy of a model on the GSM8K dataset using the eval_acc.py script.
# Usage:
#   cd /cpfs/user/fengmingquan/nanoGPT
#   nohup bash eval_acc.sh > log/eval_acc_0.out 2>&1 &
#   nohup bash eval_acc.sh > log/eval_acc_00.out 2>&1 &

# pkill -f eval_acc
# pkill -f VLLM
# fuser -v /dev/nvidia*

model_path="out-prodcpfs/qwen2-0.5B-bio-qa"  # Path to the model directory
model_name="auto"  # Specify the model name
batch_size=0  # Adjust batch size as needed, 0 for automatic selection
block_size=512  # Adjust block size as needed
n_shot_prompt=3  # Number of n-shot examples to include in the prompt
backend="vllm"  # Backend to use for evaluation, options: "hflm", "vllm", "sglang"
device="cuda"
python_path="/cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python"  # Path to the Python interpreter


#--------------
limit=2000  # Maximum number of samples to evaluate (for quick testing)
wandb_id="auto"  # Set to "auto" to automatically find the wandb ID from the log file
gpu_id_base=1
node_id=1
checkpoints=(
    40000
)
datasets=(
    "synthetic_human_age,synthetic_human_location,synthetic_human_occupation,synthetic_human_wage,synthetic_human_gender"
)

for i in "${!checkpoints[@]}"; do
    $python_path convert_pt_to_hf.py --model_path $model_path --ckpt_step ${checkpoints[$i]} --model_name $model_name
    for j in "${!datasets[@]}"; do
        dataset_name=${datasets[$j]}
        gpu_id=$((gpu_id_base + j + i))
        echo "Evaluating checkpoint ${checkpoints[$i]} on dataset $dataset_name with GPU ID $gpu_id"
        output_file="log/eval_acc_${checkpoints[$i]}_$gpu_id-$node_id.out"
        echo "Output will be saved to $output_file"
        CUDA_VISIBLE_DEVICES=$gpu_id nohup $python_path eval_acc.py --model_path $model_path --wandb_id $wandb_id --ckpt_step ${checkpoints[$i]} --model_name $model_name --batch_size $batch_size --block_size $block_size --dataset_name $dataset_name --n_shot_prompt $n_shot_prompt --limit $limit --num_proc_load_dataset 8 --device $device --backend $backend > $output_file 2>&1 &
    done
done
echo "All jobs submitted."