
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

#OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --compile=True --wandb_run_name='scrat-tinyllama-1.1B-0.4rho' --out_dir='out/scrat-tinyllama-1.1B-owm-7.5B-0.4rho' --init_from='../TinyLlama_v1.1' --ref_model_ckpt="../TinyLlama_v1.1_math_code"  --dataset='/cpfs/user/fengmingquan/dataset/processed-llama2/open-web-math' --batch_size=6 --gradient_accumulation_steps=80 --block_size=2048 --token_keep_ratio=0.4 > log/gpt-owm-rho-1.log 2>&1 &


#CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 4 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --compile=True --wandb_run_name='scrat-tinyllama-1.1B' --out_dir='out/scrat-tinyllama-1.1B-owm-15B' --init_from='TinyLlama/TinyLlama_v1.1' --ref_model_ckpt=""  --dataset='/cpfs/user/fengmingquan/dataset/processed-llama2/open-web-math' --batch_size=8 --gradient_accumulation_steps=60 --block_size=2048 --token_keep_ratio=1.0 --train_mode="resume" > log/gpt-owm-rho-2.log 2>&1 &

#OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py  --token_keep_ratio=0.4 --wandb_log=False --compile=False --out_dir='out/tmp' --init_from='Qwen/Qwen2.5-1.5B' --ref_model_ckpt="Qwen/Qwen2.5-Math-1.5B-Instruct" --dataset='/cpfs/user/fengmingquan/dataset/processed-qwen2/open-web-math' > log/gpt-owm-rho-1.log 2>&1 &

#---- single gpu cmds on DSW ----
#/cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python train.py config/cont_train_gpt2_owm_rho.py --token_keep_ratio=0.6 --wandb_run_name='cont-gpt2-0.7B-7.5B-0.6rho' --out_dir='out/cont-gpt2-0.7B-owm-7.5B-0.6rho'

# ---- multi-node cmds on DLC -----
#cd /cpfs/user/fengmingquan/nanoGPT
#OMP_NUM_THREADS=4 /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --nproc_per_node 8 --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
# train.py config/cont_train_gpt2_owm_rho.py --token_keep_ratio=0.6 --wandb_run_name='cont-gpt2-7.5B-0.6rho' --out_dir='out/cont-gpt2-124M-owm-7.5B-0.6rho'

# ---- multi-node cmds on DSW -----
#cd /cpfs/user/fengmingquan/nanoGPT
#NCCL_IB_DISABLE=1 OMP_NUM_THREADS=4 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --nproc_per_node 8 --master_addr=10.39.7.133 --master_port=12355 --nnodes=2 --node_rank=0 train.py config/cont_train_gpt2_owm_rho.py --token_keep_ratio=0.6 --wandb_run_name='cont-gpt2-1.5B-7.5B-0.6rho' --out_dir='out/cont-gpt2-1.5B-owm-7.5B-0.6rho' > log/gpt-owm-rho-0.log 2>&1 &
#NCCL_IB_DISABLE=1 OMP_NUM_THREADS=4 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --nproc_per_node 8 --master_addr=10.39.7.133 --master_port=12355 --nnodes=2 --node_rank=1 train.py config/cont_train_gpt2_owm_rho.py --token_keep_ratio=0.6 --wandb_run_name='cont-gpt2-1.5B-7.5B-0.6rho' --out_dir='out/cont-gpt2-1.5B-owm-7.5B-0.6rho' > log/gpt-owm-rho-1.log 2>&1 &

# rho-algorithm
token_keep_ratio = 0.5 

#I/O
init_from = 'gpt2-xl' # load openai pretrained gpt2 model
out_dir = 'out/cont-gpt2-0.7B-owm-15B' # output directory for checkpoints and logs
ref_model_ckpt = "out/cont-gpt2-1.5B-owm-15B/2025-07-02_21-50-52/ckpt-30000.pt"
#clustering_ckpt = "clustering_results/token_clustering_analyzer_k20.pkl"
#wandb
wandb_log = True
wandb_project = 'owm'
wandb_run_name='cont-gpt2-0.7B-owm-15B'

#data
dataset = '/cpfs/user/fengmingquan/dataset/processed-gpt2/open-web-math' # path to processed dataset

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size =  2   # 3 for 1.5B, 4 for 0.7B
block_size = 1024
gradient_accumulation_steps = 8*3*5*2

# optimizer and lr
# token per iter = 491,520
# this makes total number of tokens be 15B
max_iters = 30000
lr_decay_iters = 30000

warmup_iters = int(0.01 * max_iters) # 1% of max_iters
learning_rate = 6e-4 * 0.05 # base learning rate should be 5%
min_lr = 6e-5 * 0.1 # final learning rate should be 0.1 * base learning rate

# eval stuff
eval_interval = 2000  # frequency of eval and save.
eval_iters = 200   #loss estimation window size
log_interval = 100 # frequency of printing training status

# weight decay
weight_decay = 1e-1