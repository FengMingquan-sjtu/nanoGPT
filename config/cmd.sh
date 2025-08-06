OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 2 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=False --compile=False --wandb_run_name='cont-qwen2-0.5B-finewebedu' --out_dir='out-prodcpfs/cont-qwen2-0.5B-finewebedu' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2-0.5B' --ref_model_ckpt=""  --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --batch_size=1 --gradient_accumulation_steps=480 --block_size=4096 --token_keep_ratio=1.0 --max_iters=20000 --lr_decay_iters=20000 --learning_rate=7e-6 --min_lr=7e-7  > log/gpt-owm-rho-0.log 2>&1 &


#---- cmds -----
#CUDA_VISIBLE_DEVICES=4,5,6,7
#OMP_NUM_THREADS=4 nohup torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py > log/gpt-owm-rho.log 2>&1 &
#python train.py config/cont_train_gpt2_owm_rho.py
#compile = False # for fast try.

#---- standalone job cmds on DLC -----
#cd /cpfs/user/fengmingquan/nanoGPT
#OMP_NUM_THREADS=4 /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --token_keep_ratio=0.6 --wandb_run_name='cont-gpt2-1.5B-7.5B-0.6rho' --out_dir='out/cont-gpt2-1.5B-owm-7.5B-0.6rho'

#---- standalone job cmds on DSW -----
#OMP_NUM_THREADS=4 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --compile=True --wandb_run_name='cont-qwen2-1.5B' --out_dir='out/cont-qwen2-1.5B-owm-15B' --init_from='Qwen/Qwen2-1.5B' --ref_model_ckpt="" --dataset='/cpfs/user/fengmingquan/dataset/processed-qwen2/open-web-math' --batch_size=5 --gradient_accumulation_steps=96 --token_keep_ratio=1.0  --learning_rate=7e-6 --min_lr=7e-7 > log/gpt-owm-rho-2.log 2>&1 &

#OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --compile=True --wandb_run_name='cont-kinetgpt2-0.3B-owm-15B' --out_dir='out/cont-kinetgpt2-0.3B-owm-15B' --init_from='gpt2-medium' --ref_model_ckpt=""  --dataset='/cpfs/user/fengmingquan/dataset/processed-gpt2/open-web-math' --batch_size=15 --gradient_accumulation_steps=32 --block_size=1024 --token_keep_ratio=1.0 > log/gpt-owm-rho-1.log 2>&1 &

#OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=False --compile=True --out_dir='out/tmp' --init_from='gpt2-large' --ref_model_ckpt="out/cont-gpt2-1.5B-owm-15B/2025-07-02_21-50-52/ckpt-30000.pt"  --dataset='/cpfs/user/fengmingquan/dataset/processed-gpt2/open-web-math' --batch_size=10 --gradient_accumulation_steps=48 --block_size=1024 --token_keep_ratio=0.4 > log/gpt-owm-rho-5.log 2>&1 &

#OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python train.py config/cont_train_gpt2_owm_rho.py --wandb_log=False --compile=False --wandb_run_name='cont-tinyllama-1.1B-0.4rho-attn' --out_dir='out/cont-tinyllama-1.1B-owm-7.5B-0.4rho-attn' --init_from='../TinyLlama_v1.1' --ref_model_ckpt="../TinyLlama_v1.1_math_code"  --dataset='/cpfs/user/fengmingquan/dataset/processed-llama2/open-web-math' --batch_size=4 --gradient_accumulation_steps=120 --block_size=2048 --token_keep_ratio=0.4 --attn_select=True > log/gpt-owm-rho-3.log 2>&1 &

#OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --compile=True --wandb_run_name='cont-tinyllama-1.1B-tinypajama' --out_dir='out/cont-tinyllama-1.1B-tinypajama' --init_from='/prodcpfs/user/fengmingquan/model/TinyLlama-1.1B-intermediate-step-1431k-3T' --ref_model_ckpt=""  --dataset='/prodcpfs/user/fengmingquan/dataset/processed-llama2/tinypajama' --batch_size=6 --gradient_accumulation_steps=80 --block_size=2048 --token_keep_ratio=1.0 --max_iters=20000 --lr_decay_iters=20000  > log/gpt-owm-rho-0.log 2>&1 &

#OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=False --compile=False --wandb_run_name='cont-tinyllama-1.1B-tinypajama-0.6rho' --out_dir='out/cont-tinyllama-1.1B-tinypajama-0.6rho' --init_from='/prodcpfs/user/fengmingquan/model/TinyLlama-1.1B-intermediate-step-1431k-3T' --ref_model_ckpt="/prodcpfs/user/fengmingquan/model/TinyLlama-1.1B-Chat-v1.0"  --dataset='/prodcpfs/user/fengmingquan/dataset/processed-llama2/tinypajama' --batch_size=5 --gradient_accumulation_steps=96 --block_size=2048 --token_keep_ratio=0.6 --max_iters=20000 --lr_decay_iters=20000 > log/gpt-owm-rho-3.log 2>&1 &

#---- single gpu cmds on DSW ----
#/cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python train.py config/cont_train_gpt2_owm_rho.py --token_keep_ratio=0.6 --wandb_run_name='cont-gpt2-0.7B-7.5B-0.6rho' --out_dir='out/cont-gpt2-0.7B-owm-7.5B-0.6rho'

# ---- multi-node cmds on DLC -----
#cd /cpfs/user/fengmingquan/nanoGPT
#OMP_NUM_THREADS=4 /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --nproc_per_node 8 --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
# train.py config/cont_train_gpt2_owm_rho.py --token_keep_ratio=0.6 --wandb_run_name='cont-gpt2-7.5B-0.6rho' --out_dir='out/cont-gpt2-124M-owm-7.5B-0.6rho'
