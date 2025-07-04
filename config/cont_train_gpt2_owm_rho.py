
#---- cmds -----
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#OMP_NUM_THREADS=4 nohup torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py > log/gpt-owm-rho.log 2>&1 &
#python train.py config/cont_train_gpt2_owm_rho.py
#compile = False # for fast try.

#---- standalone job cmds on DLC -----
#cd /cpfs/user/fengmingquan/nanoGPT
#OMP_NUM_THREADS=4 /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --token_keep_ratio=0.6 --wandb_run_name='cont-gpt2-1.5B-7.5B-0.6rho' --out_dir='out/cont-gpt2-1.5B-owm-7.5B-0.6rho'

#---- standalone job cmds on DSW -----
#OMP_NUM_THREADS=4 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --token_keep_ratio=0.6 --wandb_run_name='cont-gpt2-1.5B-7.5B-0.6rho' --out_dir='out/cont-gpt2-1.5B-owm-7.5B-0.6rho' > log/gpt-owm-rho-0.log 2>&1 &

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
out_dir = 'out/cont-gpt2-1.5B-owm-15B' # output directory for checkpoints and logs
ref_model_ckpt = "out/cont-gpt2-1.5B-owm-15B/2025-07-02_21-50-52/ckpt-30000.pt"
#wandb
wandb_log = True
wandb_project = 'owm'
wandb_run_name='cont-gpt2-1.5B-owm-15B'

#data
dataset = '/cpfs/user/fengmingquan/dataset/processed-gpt2/open-web-math' # path to processed dataset

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size =  3
block_size = 1024
gradient_accumulation_steps = 8 * 5 * 4

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