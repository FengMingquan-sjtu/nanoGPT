# saves the openwebmath dataset to a binary file for training.
# Copied from ../openwebtext/prepare.py
import os
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset  # huggingface datasets
import multiprocessing as mp
# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 40

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = 40

# Load tokenizer - 根据你的需求选择合适的tokenizer
# 例如：gpt2, microsoft/DialoGPT-large, 或其他有大词汇表的模型
tokenizer = AutoTokenizer.from_pretrained("/prodcpfs/user/fengmingquan/model/Qwen2-0.5B")  # 请根据需要修改模型名称

# download data by:
input_path = "/prodcpfs/user/fengmingquan/dataset/raw/fineweb-edu/sample/10BT"
output_path = "/prodcpfs/user/fengmingquan/dataset/processed-qwen2/tmp2-fineweb-edu-10bt"

if not os.path.exists(output_path):
    os.makedirs(output_path)

def writer_process(queue, filename, dtype, shape):
    """这个函数在独立的进程中运行，专门负责从队列中获取数据并写入文件"""
    # 在子进程中重新打开 memmap 文件
    arr_writer = np.memmap(filename, dtype=dtype, mode='r+', shape=shape)
    
    idx = 0
    while True:
        # 从队列中获取数据，如果队列为空，会阻塞等待
        batch_data = queue.get()
        
        # 接收到结束信号 (None) 时，退出循环
        if batch_data is None:
            break
            
        # 写入数据
        arr_writer[idx : idx + len(batch_data)] = batch_data
        idx += len(batch_data)
        
    # 确保所有内容都写入磁盘
    arr_writer.flush()
    print("Writer process finished.")

if __name__ == '__main__':
    # 打印tokenizer信息
    print(f"Using tokenizer: {tokenizer.name_or_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")

    dataset = load_dataset(input_path, num_proc=num_proc_load_dataset)
    use_queue = False

    # filter the dataset 
    #def filter_function(example):
    #    return example['meta']["redpajama_set_name"]  == "RedPajamaWikipedia" # 23.5B tokens
    #dataset = dataset.filter(filter_function, num_proc=num_proc_load_dataset)
    
    # owt by default only contains the 'train' split, so create a test split
    if not ("val" in dataset):
        # If 'val' split does not exist, create it
        split_dataset = dataset["train"].train_test_split(test_size=0.95, seed=2357, shuffle=True) #NOTE: only keep a small portion for tmp test
        split_dataset.pop('test')
        
    else:
        split_dataset = dataset
    
    print(split_dataset)
    
    # we now want to tokenize the dataset. first define the encoding function
    def process(example):
        # Use tokenizer.encode instead of encode_ordinary
        # add_special_tokens=False is equivalent to encode_ordinary (ignores special tokens)
        ids = tokenizer.encode(example['text'], add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)  # add the end of text token
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out
    
    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    
    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(output_path, f'{split}.bin')
        
        # 修改数据类型以支持更大的词汇表
        # 检查最大token ID以确定合适的数据类型
        max_token_id = max(tokenizer.vocab_size, tokenizer.eos_token_id if tokenizer.eos_token_id else 0)
        
        if max_token_id < 2**16:
            dtype = np.uint16
            print(f"Using uint16 for vocab size {max_token_id}")
        elif max_token_id < 2**32:
            dtype = np.uint32
            print(f"Using uint32 for vocab size {max_token_id}")
        else:
            dtype = np.uint64
            print(f"Using uint64 for vocab size {max_token_id}")
        
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024
        
        if use_queue:
            queue = mp.Queue(maxsize=8) 

            # 3. 启动写入者进程
            writer = mp.Process(target=writer_process, args=(queue, filename, dtype, (arr_len,)))
            writer.start()

            # 4. 主进程作为生产者，准备数据并放入队列
            print(f"Main process started writing to {filename}")
            for batch_idx in tqdm(range(total_batches)):
                # 准备数据（这部分和你的代码一样）
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                
                # 将准备好的数据放入队列，如果队列满了，会阻塞等待
                queue.put(arr_batch)

            # 5. 所有数据都放入队列后，放入一个 "结束信号"
            queue.put(None)

            # 6. 等待写入者进程执行完毕
            writer.join()
        else:
            print(f"Single process started writing to {filename}")
            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                # Batch together samples for faster write
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                # Write into mmap
                arr[idx: idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            
            arr.flush()


        print(f"Saved {split} dataset to {filename} with {len(dset)} samples and {arr_len} tokens.")
        print(f"File size: {os.path.getsize(filename) / (1024**3):.2f} GB")
    
    # 读取文件时需要使用相同的数据类型
    print(f"\n# To read the bin files later, use the same dtype ({dtype}):")
    print(f"# m = np.memmap('train.bin', dtype=np.{dtype.__name__}, mode='r')")

# nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python data/fineweb-edu/prepare_hf_tmp.py > log/prepare_hf_04.log 2>&1 &


# fineweb-edu-10bt+qwen2 --> 9.89B tokens