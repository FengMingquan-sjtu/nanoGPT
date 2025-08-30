#nohup ./delayed_train.sh > delayed_task.log 2>&1 &
echo "$(date): 等待3.5小时后开始训练..."
sleep 12600

echo "$(date): 开始执行训练任务..."

OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --use_deepspeed=True --zero_stage=2 --wandb_run_name='qwen2-0.5B-finewebedu+nemotron' --out_dir='out-prodcpfs/qwen2-0.5B-finewebedu+nemotron' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2-0.5B' --ref_model_ckpt=''   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --train_mode='scratch' --dataset_prefix="fineweb-edu-100bt-25bt,copy-nemotron-cc-hq-smal" --dataset_ratio="50:50" --batch_size=5 --gradient_accumulation_steps=24 --block_size=4096 --token_keep_ratio=1.0 --max_iters=60000 --lr_decay_iters=60000 --learning_rate=1e-4 --min_lr=1e-5 > log/gpt-owm-rho-4.log 2>&1 &

echo "$(date): 训练任务已启动"