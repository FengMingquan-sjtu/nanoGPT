# saves a subset of openwebmath dataset to a binary file for training. 
# Copied from ./prepare.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 50

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

input_path = "/cpfs/user/fengmingquan/dataset/raw/open-web-math"
output_path = "/cpfs/user/fengmingquan/dataset/processed-gpt2/open-web-math-small"
if not os.path.exists(output_path):
    os.makedirs(output_path)

if __name__ == '__main__':
    # takes 27GB in huggingface .cache dir, about 6.32M documents
    dataset = load_dataset(input_path, num_proc=num_proc_load_dataset)

    # filter the dataset by the urls that contains one of the following keywords
    keywords = ['stackexchange.com', 'nature.com', 'wordpress.com', 'physicsforums.com',
                'github.io', 'zbmath.org', 'wikipedia.org', 'groundai.com', 'blogspot.com','mathoverflow.net']
    def filter_function(example):
        return any(keyword in example['url'] for keyword in keywords)
    dataset = dataset.filter(filter_function, num_proc=num_proc_load_dataset)

    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset["train"].train_test_split(test_size=0.001, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    print(split_dataset)

    # DatasetDict({
    #     train: Dataset({
    #         features: ['url', 'text', 'date', 'metadata'],
    #         num_rows: 1969492
    #     })
    #     val: Dataset({
    #         features: ['url', 'text', 'date', 'metadata'],
    #         num_rows: 1972
    #     })
    # })


    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
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
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"Saved {split} dataset to {filename} with {len(dset)} samples and {arr_len} tokens.")
    # train.bin is ~6.6GB, val.bin ~6.8MB
    # train has ~3.5B tokens (3,491,845,094)
    # val has ~3M tokens (3,560,451)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
