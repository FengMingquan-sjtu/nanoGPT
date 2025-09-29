python_path="/cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python"  # Path to the Python interpreter
model_names="/prodcpfs/user/fengmingquan/model/Qwen2-0.5B;/prodcpfs/user/fengmingquan/model/Qwen2-1.5B"
top_k_list=(3 5 10 20 30 40 50 60)
gpu_id_list=(0 1 2 3 4 5 6 7)
for i in "${gpu_id_list[@]}"; do
    nohup $python_path print_model.py \
        --model_names $model_names \
        --k ${top_k_list[$i]} \
        --gpu_id ${gpu_id_list[$i]} \
        > log/print_model_${i}.out 2>&1 &
done