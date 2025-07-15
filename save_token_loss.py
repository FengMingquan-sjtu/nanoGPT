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
from torch.nn import functional as F
from model import GPTConfig, GPT

# ---- tunable model config ----
block_size = 1024
batch_size = 24
dataset = '/cpfs/user/fengmingquan/dataset/processed-gpt2/open-web-math'
ref_model_ckpt = "out/cont-gpt2-1.5B-owm-15B/2025-07-02_21-50-52/ckpt-30000.pt"
output_file = os.path.join(dataset, "gpt2-1.5B-owm-token-losses-train.bin")
split = 'train' # 'train' or 'val', specify which split to compute losses for

# ---- model config ----
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ---- DDP setup ----
def setup_ddp():
    """初始化DDD环境"""
    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def cleanup_ddp():
    """清理DDP环境"""
    destroy_process_group()

# ---- data loading ----
data_dir = dataset

def get_batch_indices_with_remainder(split, start_idx, end_idx, data_len):
    """获取指定范围的批次数据，包括处理剩余tokens"""
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # 计算完整blocks的数量
    total_complete_samples = (data_len - 1) // block_size
    remainder_tokens = (data_len - 1) % block_size
    
    print(f"Total data length: {data_len}, Complete samples: {total_complete_samples}, Remainder tokens: {remainder_tokens}")
    
    # 收集所有要处理的样本信息 (sample_idx, actual_length)
    samples_to_process = []
    
    # 处理完整的blocks
    for idx in range(start_idx, min(end_idx, total_complete_samples)):
        samples_to_process.append((idx, block_size))
    
    # 处理剩余的tokens（只有最后一个进程处理）
    if remainder_tokens > 0 and end_idx > total_complete_samples:
        samples_to_process.append((total_complete_samples, remainder_tokens))
        print(f"Process handling remainder: {remainder_tokens} tokens")
    
    # 分批处理
    batches = []
    for i in range(0, len(samples_to_process), batch_size):
        batch_samples = samples_to_process[i:i+batch_size]
        if len(batch_samples) == 0:
            break
        
        # 获取这个批次的最大长度
        max_length = max(length for _, length in batch_samples)
        
        batch_x = []
        batch_y = []
        batch_info = []  # (sample_idx, actual_length)
        
        for sample_idx, actual_length in batch_samples:
            if actual_length == block_size:
                # 完整的block
                x_seq = torch.from_numpy((data[sample_idx*block_size:(sample_idx+1)*block_size]).astype(np.int64))
                y_seq = torch.from_numpy((data[sample_idx*block_size+1:(sample_idx+1)*block_size+1]).astype(np.int64))
            else:
                # 剩余的tokens
                start_pos = sample_idx * block_size
                x_seq = torch.from_numpy((data[start_pos:start_pos+actual_length]).astype(np.int64))
                y_seq = torch.from_numpy((data[start_pos+1:start_pos+actual_length+1]).astype(np.int64))
                
                # 如果需要padding到统一长度（用于批处理）
                if len(x_seq) < max_length:
                    pad_length = max_length - len(x_seq)
                    x_seq = torch.cat([x_seq, torch.zeros(pad_length, dtype=torch.int64)])
                    y_seq = torch.cat([y_seq, torch.full((pad_length,), -100, dtype=torch.int64)])  # 使用ignore_index padding
            
            batch_x.append(x_seq)
            batch_y.append(y_seq)
            batch_info.append((sample_idx, actual_length))
        
        x = torch.stack(batch_x)
        y = torch.stack(batch_y)
        
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
            
        batches.append((x, y, batch_info))
    
    return batches

def save_losses_to_binary_with_remainder(losses_dict, output_path, data_len):
    """将losses保存为二进制文件，处理不同长度的序列"""
    # 按索引排序
    sorted_items = sorted(losses_dict.items())
    
    # 计算总的token数量
    total_tokens = data_len - 1  # 减1是因为最后一个token没有对应的target
    losses_array = np.full(total_tokens, np.nan, dtype=np.float32)  # 用NaN初始化，便于检查
    
    current_pos = 0
    for sample_idx, sample_losses in sorted_items:
        actual_length = len(sample_losses)
        losses_array[current_pos:current_pos+actual_length] = sample_losses
        current_pos += actual_length
    
    # 检查是否有遗漏的tokens
    nan_count = np.isnan(losses_array).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} tokens were not processed!")
    
    # 使用memmap保存
    output_memmap = np.memmap(output_path, dtype=np.float32, mode='w+', shape=(total_tokens,))
    output_memmap[:] = losses_array[:]
    output_memmap.flush()
    del output_memmap
    
    print(f"Saved token losses to {output_path}, total tokens: {total_tokens}")
    return total_tokens

def gather_losses_efficiently(local_losses, world_size, global_rank):
    """高效收集所有进程的losses"""
    if global_rank == 0:
        all_losses = local_losses.copy()
        
        for rank in range(1, world_size):
            # 接收数据
            recv_buffer = [None]
            torch.distributed.recv_object_list(recv_buffer, src=rank)
            rank_losses = recv_buffer[0]
            
            if rank_losses:
                all_losses.update(rank_losses)
                
        return all_losses
    else:
        # 发送数据到主进程
        torch.distributed.send_object_list([local_losses], dst=0)
        return None

def main():
    # 初始化DDP
    setup_ddp()
    
    # 获取DDP信息
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    print(f"Process {global_rank}/{world_size} (local rank {local_rank}) starting...")
    
    # 加载模型
    print(f"Loading reference model from {ref_model_ckpt}")
    if ref_model_ckpt.endswith('.pt'):
        ref_checkpoint = torch.load(ref_model_ckpt, map_location=device)
        ref_model_args = ref_checkpoint['model_args']
        ref_gptconf = GPTConfig(**ref_model_args)
        ref_model = GPT(ref_gptconf)
        ref_state_dict = ref_checkpoint['model']
        ref_model.load_ckp_state_dict(ref_state_dict)
    else:
        override_args = dict(dropout=0.0)
        ref_model = GPT.from_pretrained(ref_model_ckpt, override_args)
    
    ref_model.to(device)
    ref_model.eval()
    
    # 编译和包装模型
    ref_model = torch.compile(ref_model)
    ref_model = DDP(ref_model, device_ids=[local_rank])
    
    # 计算数据分割 
    data = np.memmap(os.path.join(data_dir, split+'.bin'), dtype=np.uint16, mode='r')
    data_len = len(data)
    total_complete_samples = (data_len - 1) // block_size
    remainder_tokens = (data_len - 1) % block_size
    
    print(f"Data length: {data_len}, Complete samples: {total_complete_samples}, Remainder: {remainder_tokens}")
    
    # 按进程分割数据 - 包括剩余部分
    total_samples_to_process = total_complete_samples + (1 if remainder_tokens > 0 else 0)
    samples_per_process = total_samples_to_process // world_size
    start_idx = global_rank * samples_per_process
    
    if global_rank == world_size - 1:  # 最后一个进程处理剩余的样本
        end_idx = total_samples_to_process
    else:
        end_idx = start_idx + samples_per_process
    
    print(f"Process {global_rank} processing samples {start_idx} to {end_idx-1}")
    
    # 获取当前进程需要处理的批次
    batches = get_batch_indices_with_remainder(split, start_idx, end_idx, data_len)
    
    # 计算token losses
    ignore_idx = getattr(ref_model.module.config, 'ignore_index', -100)
    local_losses = {}
    
    print(f"Process {global_rank} processing {len(batches)} batches...")
    
    for batch_idx, (x, y, batch_info) in enumerate(batches):
        if batch_idx % 10 == 0:
            print(f"Process {global_rank} processing batch {batch_idx}/{len(batches)}")
            
        with torch.no_grad():
            with ctx:
                logits, _ = ref_model(x, y)
                
            b, t, vocab_size = logits.size()
            token_loss = F.cross_entropy(
                logits.view(-1, vocab_size), 
                y.view(-1), 
                ignore_index=ignore_idx, 
                reduction='none'
            )
            token_loss = token_loss.view(b, t)  # shape (b, t)
            
            # 修复：先转换为float32再转为numpy
            if token_loss.dtype in [torch.bfloat16, torch.float16]:
                token_loss = token_loss.float()  # 转换为float32
            
            token_loss = token_loss.cpu().numpy()  # 现在可以安全转换为numpy
            
            # 存储每个样本的losses，只保留有效部分
            for i, (sample_idx, actual_length) in enumerate(batch_info):
                # 只保留实际长度的losses，忽略padding部分
                valid_losses = token_loss[i, :actual_length].astype(np.float32)
                local_losses[sample_idx] = valid_losses
    
    print(f"Process {global_rank} finished computing losses for {len(local_losses)} samples")
    
    # 收集所有进程的结果
    torch.distributed.barrier()  # 同步所有进程
    
    # 使用更高效的对象传输方式
    all_losses = gather_losses_efficiently(local_losses, world_size, global_rank)
    
    if global_rank == 0:
        print(f"Master process collected losses for {len(all_losses)} total samples")
        
        # 保存结果
        total_saved = save_losses_to_binary_with_remainder(all_losses, output_file, data_len)
        print(f"Token losses saved successfully! Total tokens saved: {total_saved}")
        print(f"Original data length: {data_len}, Saved tokens: {total_saved}")
    
    # 清理
    cleanup_ddp()
    print(f"Process {global_rank} finished!")

if __name__ == "__main__":
    main()
    #nohup torchrun --nproc_per_node=8 save_token_loss.py > log/gpt-owm-rho-save-loss.log 2>&1 &