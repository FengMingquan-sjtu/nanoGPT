echo "$(date): 等待2小时后开始训练..."
sleep 7

echo "$(date): 开始执行训练任务..."

OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --use_deepspeed=True --zero_stage=2 --wandb_run_name='qwen2-0.5B-finewebedu+cosmopedia' --out_dir='out-prodcpfs/qwen2-0.5B-finewebedu+cosmopedia' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2-0.5B' --ref_model_ckpt=''   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --train_mode='scratch' --dataset_prefix="fineweb-edu-100bt-25bt,cosmopedia-v2" --dataset_ratio="50:50" --batch_size=20 --gradient_accumulation_steps=24 --block_size=1024 --token_keep_ratio=1.0 --max_iters=40000 --lr_decay_iters=40000 --learning_rate=1e-4 --min_lr=1e-5  > log/gpt-owm-rho-4.log 2>&1 &

echo "$(date): 训练任务已启动"