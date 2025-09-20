# saves the openwebmath dataset to a binary file for training.
# Copied from ../openwebtext/prepare.py
import os
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset  # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 40

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = 40

# Load tokenizer - 根据你的需求选择合适的tokenizer
# 例如：gpt2, microsoft/DialoGPT-large, 或其他有大词汇表的模型
#tokenizer = AutoTokenizer.from_pretrained("/prodcpfs/user/fengmingquan/model/Qwen2-0.5B")  # 请根据需要修改模型名称

from hf_olmo import OLMoTokenizerFast
tokenizer = OLMoTokenizerFast.from_pretrained("/prodcpfs/user/fengmingquan/model/OLMo-1B", trust_remote_code=True)

# download data by:
#input_path = "/prodcpfs/user/fengmingquan/dataset/raw/fineweb-edu/sample/10BT"
#output_path = "/prodcpfs/user/fengmingquan/dataset/processed-qwen2/fineweb-edu-10bt"
#input_path = "/prodcpfs/user/fengmingquan/dataset/raw/fineweb-edu/sample/100BT-25BT"
#output_path = "/prodcpfs/user/fengmingquan/dataset/processed-qwen2/fineweb-edu-100bt-25bt"
input_path = "/oss/crawl/multimodal/HuggingFaceFW/fineweb-edu/sample/100BT/"
#output_path = "/prodcpfs/user/fengmingquan/dataset/processed-qwen2/fineweb-edu-100bt-20bt"
output_path = "/prodcpfs/user/fengmingquan/dataset/processed-olmo/fineweb-edu-100bt-17bt"
total_batches = 1024  # 1024 for 10BT, 2048 for 25BT, 4096 for 100BT

if not os.path.exists(output_path):
    os.makedirs(output_path)

if __name__ == '__main__':
    # 打印tokenizer信息
    print(f"Using tokenizer: {tokenizer.name_or_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")

    #dataset = load_dataset(input_path, num_proc=num_proc_load_dataset)
    if output_path.endswith('fineweb-edu-100bt-20bt'):
        start, end = 75, 100  # 75~100
    elif output_path.endswith('fineweb-edu-100bt-19bt'):
        start, end = 50, 75  # 50~75
    elif output_path.endswith('fineweb-edu-100bt-18bt'):
        start, end = 25, 50  # 25~50
    elif output_path.endswith('fineweb-edu-100bt-17bt'):
        start, end = 0, 25  # 00~25
    else:
        raise ValueError("Please set the correct output_path ending to determine the data range to load.")
    data_files = []
    for idx in range(start, end):  # 00~25, 25~50, 50~75, 75~100
        i = idx // 10
        j = idx % 10
        data_files.append(os.path.join(input_path, f'00{i}_0000{j}.parquet'))
    print(f"Loading {len(data_files)} files...")
    print(data_files)
    dataset = load_dataset("parquet", data_files=data_files, num_proc=num_proc_load_dataset)

    # filter the dataset 
    #def filter_function(example):
    #    return example['meta']["redpajama_set_name"]  == "RedPajamaWikipedia" # 23.5B tokens
    #dataset = dataset.filter(filter_function, num_proc=num_proc_load_dataset)
    
    # owt by default only contains the 'train' split, so create a test split
    if not ("val" in dataset):
        # If 'val' split does not exist, create it
        split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
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

# nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python data/fineweb-edu/prepare_hf.py > log/prepare_hf_04.log 2>&1 &
# prepare_hf_01.log -- 17bt --- 00~25.parquet
# prepare_hf_02.log -- 18bt --- 25~50.parquet
# prepare_hf_03.log -- 19bt --- 50~75.parquet
# prepare_hf_04.log -- 20bt --- 75~100.parquet


# fineweb-edu-10bt+qwen2 --> 9.89B tokens
# processed-qwen2/fineweb-edu-100bt-25bt/train.bin with 24220569 samples and 24721353842 tokens. (24.72 B tokens)

# processed-qwen2/fineweb-edu-100bt-17bt/train.bin with 18198896 samples and 18716911567 tokens. (18.72 B tokens)
# processed-qwen2/fineweb-edu-100bt-18bt/train.bin with 18415787 samples and 18664843604 tokens.
# processed-qwen2/fineweb-edu-100bt-19bt/train.bin with 18339825 samples and 18748648871 tokens.
# processed-qwen2/fineweb-edu-100bt-20bt/train.bin with 18046972 samples and 18753313740 tokens.