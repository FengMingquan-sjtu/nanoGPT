#!/bin/bash
# This script evaluates the accuracy of a model on the GSM8K dataset using the eval_acc.py script.
# Usage:
#   cd /cpfs/user/fengmingquan/nanoGPT
#   nohup bash eval_acc.sh > log/eval_acc_0.out 2>&1 &
#   nohup bash eval_acc.sh > log/eval_acc_00.out 2>&1 &
#   nohup bash eval_acc.sh > log/eval_acc_000.out 2>&1 &
#   nohup bash eval_acc.sh > log/eval_acc_0000.out 2>&1 &

# pkill -f eval_acc
# pkill -f VLLM
# fuser -v /dev/nvidia*

model_path="out-prodcpfs/qwen2-0.5B-finewebedu-distil-2.0-0.9-top50"  #+cosmopedia  +nemotron  -distil-2.0-0.9-0.9rho  -distil-2.0-0.9-top50
#model_path="/prodcpfs/user/fengmingquan/model/Qwen2-0.5B"
model_name="auto"  # Specify the model name
batch_size=0  # Adjust batch size as needed, 0 for automatic selection
block_size=4096  # Adjust block size as needed

backend="vllm"  # Backend to use for evaluation, options: "hflm", "vllm", "sglang"
device="cuda"
python_path="/cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python"  # Path to the Python interpreter
gpu_ratio=0.8  

#--------------
limit=1000000  # Maximum number of samples to evaluate (for quick testing)
wandb_id="auto"  # Set to "auto" to automatically find the wandb ID from the log file
gpu_id_base=2
node_id=0

n_shot_prompt=5  #ppl task use 0 shot; acc task use 5 shot


checkpoints=(
#    26000 30000 36000 40000 50000 60000 70000 78000
#    4000 8000 10000 20000 30000 40000 50000 60000
#     40000 50000 60000 70000 80000 90000 100000 110000
#     64000 70000 74000 80000 84000 90000 94000 100000
#     2000 4000 8000 12000 16000 20000 30000 40000
#     50000 60000 70000 80000 90000 100000 110000 120000
#    130000 140000 150000 160000 170000 180000 190000 200000
    200000 210000
#    206000 210000
)
datasets=(
#    "mmlu,arc_challenge,arc_easy,hellaswag,winogrande,mbpp,humaneval,gsm8k,gpqa_main_n_shot"
#     "gsm8k,mmlu_pro"
#    "arc_challenge,arc_easy,hellaswag,winogrande,piqa,openbookqa"
    "hellaswag"
     "arc_challenge,arc_easy"
     "winogrande,piqa,openbookqa"
#    "c4,pile_10k,wikitext"  # need hflm backend
#    "c4"
#    "pile_10k"
#    "wikitext"
#   "hellaswag_gpt_ppl,arc_challenge_gpt_ppl" 
#    "hellaswag_ppl"
#    "hellaswag_gpt_ppl"
#    "hellaswag_gpt_ppl,arc_challenge_gpt_ppl" 
)
# mmlu,mmlu_pro,arc_challenge,arc_easy,gpqa_main_n_shot
# hellaswag,winogrande,mbpp
# ----
# drop
# humaneval,mbpp
# hendrycks_math,gsm8k
for i in "${!checkpoints[@]}"; do
    $python_path convert_pt_to_hf.py --model_path $model_path --ckpt_step ${checkpoints[$i]} --model_name $model_name
    for j in "${!datasets[@]}"; do
        dataset_name=${datasets[$j]}
        gpu_id=$((gpu_id_base + j + i))
        echo "Evaluating checkpoint ${checkpoints[$i]} on dataset $dataset_name with GPU ID $gpu_id"
        output_file="log/eval_acc_${checkpoints[$i]}_$gpu_id-$node_id.out"
        echo "Output will be saved to $output_file"
        CUDA_VISIBLE_DEVICES=$gpu_id nohup $python_path eval_acc.py --model_path $model_path --wandb_id $wandb_id --ckpt_step ${checkpoints[$i]} --model_name $model_name --batch_size $batch_size --block_size $block_size --dataset_name $dataset_name --n_shot_prompt $n_shot_prompt --limit $limit --num_proc_load_dataset 8 --device $device --backend $backend --gpu_ratio $gpu_ratio > $output_file 2>&1 &
    done
done
echo "All jobs submitted."


# (${winogrande/acc,none}+${piqa/acc_norm,none}+${openbookqa/acc_norm,none}+${arc_easy/acc_norm,none}+${arc_challenge/acc_norm,none}+${hellaswag/acc_norm,none})/6