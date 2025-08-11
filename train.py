"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""
import sys
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F

from model import GPTConfig, GPT
from model import get_loss_rho, get_loss_cls_rho, remove_prefix_from_state_dict
from model import configure_AdamW_optimizer
from token_cluster import TokenClusteringAnalyzer, GPTFeatureExtractor

os.environ["WANDB_API_KEY"] = "b7f26328382adc825eb193aac3f30f07e7da99c1" # set your wandb api key here
rank = int(os.environ.get('RANK', 0))
#os.environ['TRITON_CACHE_DIR'] = f'/prodcpfs/user/fengmingquan/triton_cache/rank_{rank}'
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# rho algorithm
token_keep_ratio = 0.5 # keep 50% of tokens, i.e. 1/2 of the original dataset

# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = ''  # model name or path
train_mode = 'cont_pretrain' # 'scratch' or 'resume' or 'cont_pretrain'
# token selection
ref_model_ckpt = "" # reference model file, used for data selection
ref_model_ckpt_2 = "" # second reference model file, used for data selection
clustering_ckpt = "" # token clustering results, used for cluster data selection
reverse_select = False # if True, select tokens in reverse order.
scale_select = False # if True, select tokens by relative scale of loss instead of difference.
batch_select = False # if True, select tokens in batches, otherwise select tokens in sample.
mask_select = 0 # if > 0, use attention mask to select tokens, otherwise use no mask.
value_select = False # if True, use the value of the loss instead of the difference from the reference model.
smooth_kernel_size = 1 # kernel size for smoothing the token loss, 1 means no smoothing
attn_select = False # 
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = '/prodcpfs/user/fengmingquan/dataset/processed-qwen2' #root path to processed dataset
dataset_prefix = "fineweb-edu,megamath,opc-ann" # prefixs of the dataset to use, separated by commas
dataset_ratio = "50,25,25" # ratio of the dataset to use, separated by commas
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
use_muon=False
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
if int(os.environ.get('RANK', -1)) <= 0:
    if train_mode != 'resume':
        out_folder = os.path.join(out_dir, f'{time.strftime("%Y-%m-%d_%H-%M-%S")}')
        os.makedirs(out_folder)
    else:
        # find the latest folder in the out_dir by sorting the folder names
        folders = [f for f in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, f)) and f.startswith('20')]
        out_folder = os.path.join(out_dir, sorted(folders)[-1])
    print(f"Output directory: {out_folder}")
    out_file = os.path.join(out_folder, 'out.log')
    sys.stdout = open(out_file, 'a', buffering=30)
    sys.stderr = open(out_file, 'a', buffering=30)
    print("configs are:", config, flush=True)
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    print(f"Running in DDP mode with rank {ddp_rank} (local {ddp_local_rank}) and world size {ddp_world_size}")
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")


torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = dataset
def get_batch(split):
    if "qwen2" in dataset:
        data_dtype = np.uint32
    else:
        data_dtype = np.uint16

    x_, y_ = [], []
    for i in range(batch_size):
        dataset_freq = np.array([float(x.strip()) for x in dataset_ratio.split(',')])
        dataset_idx = np.random.choice(len(dataset_freq), p=dataset_freq/dataset_freq.sum())
        dataset_prefix_i = dataset_prefix.split(",")[dataset_idx]
        datasubset_folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith(dataset_prefix_i)]
        datasubset_freq = np.array([os.path.getsize(os.path.join(f, f'train.bin')) for f in datasubset_folders])
        datasubset_idx = np.random.choice(len(datasubset_freq), p=datasubset_freq/datasubset_freq.sum())
        datasubset_folder_i = datasubset_folders[datasubset_idx]
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(os.path.join(datasubset_folder_i, 'train.bin'), dtype=data_dtype, mode='r')
        else:
            data = np.memmap(os.path.join(datasubset_folder_i, 'val.bin'), dtype=data_dtype, mode='r')
        ix = torch.randint(len(data) - block_size, (1,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x_.append(x)
        y_.append(y)
    x = torch.cat(x_, dim=0)
    y = torch.cat(y_, dim=0)
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # NOTE: depracated. only for ckpt format compatibility
#if init_from == 'scratch':
#    # init a new model from scratch
#    print("Initializing a new model from scratch")
#    # determine the vocab size we'll use for from-scratch training
#    if meta_vocab_size is None:
#        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
#    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
#    gptconf = GPTConfig(**model_args)
#    model = GPT(gptconf)
#elif init_from == 'resume':
#    print(f"Resuming training from {out_dir}")
#    # resume training from a checkpoint.
#    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
#    checkpoint = torch.load(ckpt_path, map_location=device)
#    checkpoint_model_args = checkpoint['model_args']
#    # force these config attributes to be equal otherwise we can't even resume training
#    # the rest of the attributes (e.g. dropout) can stay as desired from command line
#    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
#        model_args[k] = checkpoint_model_args[k]
#    # create the model
#    gptconf = GPTConfig(**model_args)
#    model = GPT(gptconf)
#    state_dict = checkpoint['model']
#    model.load_ckp_state_dict(state_dict)
#    iter_num = checkpoint['iter_num']
#    best_val_loss = checkpoint['best_val_loss']
#elif init_from.startswith('gpt2'):
#    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
#    # initialize from OpenAI GPT-2 weights
#    override_args = dict(dropout=dropout)
#    model = GPT.from_pretrained(init_from, override_args)
#    # read off the created config params, so we can store them into checkpoint correctly
#    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
#        model_args[k] = getattr(model.config, k)
## crop down the model block size if desired, using model surgery
#if block_size < model.config.block_size:
#    model.crop_block_size(block_size)
#    model_args['block_size'] = block_size # so that the checkpoint will have the right value

if "gpt" in init_from:
    
    
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
else:
    if train_mode == "cont_pretrain":
        model = AutoModelForCausalLM.from_pretrained(init_from)
        print(f"Loaded main model from {init_from}")
    elif train_mode == "scratch":
        model_config = AutoConfig.from_pretrained(init_from)
        model = AutoModelForCausalLM.from_config(model_config)
        print(f"WARNING: Loaded initialized model from {init_from} with config: {model_config}")
    elif train_mode == "resume":
        folders = [f for f in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, f)) and f.startswith('20')]
        out_folder = os.path.join(out_dir, sorted(folders)[-1])
        print(f"Resuming training from {out_folder}")
        logfile = os.path.join(out_folder, "out.log")
        with open(logfile, 'r') as f:
            for line in f:
                if "wandb:" in line and "/runs/" in line:
                    wandb_id = line.split("/runs/")[-1].strip()
                    break
        ckpts = [f for f in os.listdir(out_folder) if f.startswith('ckpt') and f.endswith('.pt')]
        ckpt_path = os.path.join(out_folder, sorted(ckpts)[-1])
        checkpoint = torch.load(ckpt_path, map_location=device)
        model_config = AutoConfig.from_pretrained(init_from)
        model = AutoModelForCausalLM.from_pretrained(init_from, config=model_config)
        state_dict = checkpoint['model']
        state_dict = remove_prefix_from_state_dict(state_dict)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
model.to(device)




# load reference model checkpoint, used for data selection
ref_model = None
#if len(ref_model_ckpt)>0 and mask_select==0:
#    print(f"Loading reference model from {ref_model_ckpt}")
#    if ref_model_ckpt.endswith('.pt'):
#        # resume training from a checkpoint.
#        ref_checkpoint = torch.load(ref_model_ckpt, map_location=device)
#        ref_model_args = ref_checkpoint['model_args']
#        # create the model
#        ref_gptconf = GPTConfig(**ref_model_args)
#        ref_model = GPT(ref_gptconf)
#        ref_state_dict = ref_checkpoint['model']
#        ref_model.load_ckp_state_dict(ref_state_dict)
#    else:
#        # load from a pretrained model
#        override_args = dict(dropout=dropout)
#        ref_model = GPT.from_pretrained(ref_model_ckpt, override_args)
#    ref_model.to(device)
#    ref_model.eval()

if len(ref_model_ckpt)>0 and mask_select==0:
    if "gpt2" in ref_model_ckpt:
        if ref_model_ckpt.endswith('.pt'):
            # resume training from a checkpoint.
            ref_checkpoint = torch.load(ref_model_ckpt, map_location=device)
            ref_model_args = ref_checkpoint['model_args']
            # create the model
            ref_gptconf = GPTConfig(**ref_model_args)
            ref_model = GPT(ref_gptconf)
            ref_state_dict = ref_checkpoint['model']
            ref_model.load_ckp_state_dict(ref_state_dict)
        else:
            # load from a pretrained model
            override_args = dict(dropout=dropout)
            ref_model = GPT.from_pretrained(ref_model_ckpt, override_args)
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_ckpt)
    ref_model.to(device)
    ref_model.eval()
    print(f"Loaded reference model from {ref_model_ckpt}")

ref_model_2 = None


# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler("cuda", enabled=(dtype == 'float16'))

# optimizer
if not use_muon:
    optimizer = configure_AdamW_optimizer(model, weight_decay, learning_rate, (beta1, beta2), device_type)
else:
    print("Using Muon optimizer")
    optimizer = model.configure_muon_optimizer(weight_decay, learning_rate, (beta1, beta2), device_type)
if train_mode == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory


# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0
    if ref_model is not None:
        if len(clustering_ckpt)==0:
            # compile the reference model as well, if it exists. 
            ref_model = torch.compile(ref_model)
        else:
            #BUG: compile does not work with hooks yet, so we cannot compile the reference model if we use clustering
            print("WARNING: clustering_ckpt is set, but reference model is not compiled, because hooks are not supported in compiled models yet.")
    if ref_model_2 is not None:
        ref_model_2 = torch.compile(ref_model_2)

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    if ref_model is not None:
        # wrap the reference model into DDP container as well, if it exists
        ref_model = DDP(ref_model, device_ids=[ddp_local_rank])
    if ref_model_2 is not None:
        # wrap the second reference model into DDP container as well, if it exists
        ref_model_2 = DDP(ref_model_2, device_ids=[ddp_local_rank])

if mask_select != 0:
    ref_model = model

# load token clustering results, used for cluster data selection
# cluster analyzer
cluster_analyzer = None
feature_extractor = None
if len(clustering_ckpt)>0 and ref_model is not None:
    cluster_analyzer = TokenClusteringAnalyzer.load_state(clustering_ckpt) # load scalar+pca+kmeans cluster model.
    feature_extractor = GPTFeatureExtractor(ref_model)  #register forward hook on ref_model

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                if init_from.startswith('gpt2'):
                    logits, loss = model(X, Y)
                else:
                    logits = model(X).logits
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), reduction='mean')
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    if train_mode == 'resume':
        wandb.init(id=wandb_id, resume='must', project=wandb_project, config=config)
    else:
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        if use_muon and param_group["use_muon"]:
            frac = min(iter_num / 300, 1) # momentum warmup for muon
            param_group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    # evaluate the loss on train/val sets and write checkpoints
    if (iter_num % eval_interval == 0 or iter_num==eval_interval//2) and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }, step=iter_num)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_folder}")
                torch.save(checkpoint, os.path.join(out_folder, f'ckpt-{iter_num}.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            model.train() 
            if init_from.startswith('gpt2'):
                logits, _ = model(X, Y)
            else:
                logits = model(X).logits
            
            loss = get_loss_rho(logits, Y, ref_model, X, token_keep_ratio, reverse_select, batch_select, scale_select, mask_select, value_select, ref_model_2, smooth_kernel_size, attn_select)
            #if len(clustering_ckpt)==0:
            #    loss = get_loss_rho(logits, Y, ref_model, X, token_keep_ratio, reverse_select, batch_select, scale_select, mask_select, value_select, ref_model_2, smooth_kernel_size)
            #else:
            #    loss = get_loss_cls_rho(logits, Y, ref_model, X, token_keep_ratio, cluster_analyzer, feature_extractor, reverse_select)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            #mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            #running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            running_mfu = 0
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, remaining {(max_iters - iter_num) * dt / 3600:.2f}h", flush=True)
        if wandb_log: wandb.log({"train/loss-single-step": lossf,}, step=iter_num)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
