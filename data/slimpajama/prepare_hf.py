# saves the openwebmath dataset to a binary file for training.
# Copied from ../openwebtext/prepare.py
import os
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset  # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 80

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

# Load tokenizer - 根据你的需求选择合适的tokenizer
# 例如：gpt2, microsoft/DialoGPT-large, 或其他有大词汇表的模型
tokenizer = AutoTokenizer.from_pretrained("/cpfs/user/fengmingquan/TinyLlama_v1.1")  # 请根据需要修改模型名称

input_path = "/cpfs/user/fengmingquan/dataset/raw/slimpajama"
output_path = "/cpfs/user/fengmingquan/dataset/processed-llama2/slimpajama"

if not os.path.exists(output_path):
    os.makedirs(output_path)

if __name__ == '__main__':
    # 打印tokenizer信息
    print(f"Using tokenizer: {tokenizer.name_or_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    
    # takes 27GB in huggingface .cache dir, about 6.32M documents
    dataset = load_dataset(input_path, num_proc=num_proc_load_dataset)

    # filter the dataset by the urls that contains one of the following keywords
    #keywords = ['stackexchange.com', 'nature.com', 'wordpress.com', 'physicsforums.com',
    #            'github.io', 'zbmath.org', 'wikipedia.org', 'groundai.com', 'blogspot.com','mathoverflow.net']
    #def filter_function(example):
    #    return any(keyword in example['url'] for keyword in keywords)
    #dataset = dataset.filter(filter_function, num_proc=num_proc_load_dataset)
    
    # owt by default only contains the 'train' split, so create a test split
    if not ("val" in dataset):
        # If 'val' split does not exist, create it
        split_dataset = dataset["train"].train_test_split(test_size=0.001, seed=2357, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test')
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

# nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python data/slimpajama/prepare_hf.py > log/prepare_hf.log 2>&1 &