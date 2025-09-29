model_path="out-prodcpfs/qwen2-0.5B-finewebedu-distil-2.0-0.9-top50;out-prodcpfs/qwen2-0.5B-finewebedu" 
model_name="auto"
wandb_id="auto"
metric_name="arc_challenge_gpt/bits_per_byte,none;hellaswag_gpt/bits_per_byte,none"

python_path="/cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python" 

nohup $python_path draw_ppl_curve.py \
    --model_path $model_path \
    --model_name $model_name \
    --wandb_id $wandb_id \
    --metric_name $metric_name \
    > log/draw_ppl.out 2>&1 &