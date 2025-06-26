
#cmds
#OMP_NUM_THREADS=8 nohup torchrun --standalone --nproc_per_node=8 train.py config/cont_train_gpt2_owm_rho.py > log/owm-rho.log 2>&1 &
#python train.py config/cont_train_gpt2_owm_rho.py
#compile = False # for fast try.

#I/O
init_from = 'gpt2' # load openai pretrained gpt2 model
out_dir = 'out/cont-gpt2-124M-owm-15B-rho' # output directory for checkpoints and logs
ref_model_ckpt = 'out/cont-gpt2-owm-37B/ckpt.pt'
#wandb
wandb_log = True
wandb_project = 'owm'
wandb_run_name='cont-gpt2-15B-rho'

#data
dataset = '/cpfs/user/fengmingquan/dataset/processed-gpt2/open-web-math' # path to processed dataset

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12 * 5
block_size = 1024
gradient_accumulation_steps = 8

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