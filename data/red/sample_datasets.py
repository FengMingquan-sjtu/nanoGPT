#!/usr/bin/env python3
"""
从数据集中读取前N条样本并保存到txt文件
"""

import os
import glob
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def sample_first_n_data(dataset_dir, tokenizer_path, num_samples=10, output_file=None):
    """
    从数据集中读取前N条样本并保存到txt文件
    
    Args:
        dataset_dir: 数据集目录路径
        tokenizer_path: tokenizer 路径
        num_samples: 要读取的样本数量
        output_file: 输出文件路径（如果为None，则打印到控制台）
    """
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    print(f"\nLoading dataset from: {dataset_dir}")
    
    # 查找所有 parquet 文件
    parquet_files = sorted(glob.glob(os.path.join(dataset_dir, "*.parquet")))
    
    if not parquet_files:
        print(f"Error: No parquet files found in {dataset_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # 准备输出
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        f = open(output_file, 'w', encoding='utf-8')
        print(f"\nWriting to: {output_file}")
    else:
        f = None
    
    def write_line(line):
        if f:
            f.write(line + '\n')
        else:
            print(line)
    
    # 写入头部信息
    write_line("=" * 80)
    write_line(f"First {num_samples} Samples from: {os.path.basename(dataset_dir)}")
    write_line("=" * 80)
    write_line("")
    
    samples_collected = 0
    
    for file_idx, file_path in enumerate(parquet_files):
        if samples_collected >= num_samples:
            break
        
        print(f"Reading file {file_idx + 1}/{len(parquet_files)}: {os.path.basename(file_path)}")
        
        # 读取当前文件
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        print(f"  Rows in this file: {len(df)}")
        
        # 遍历当前文件中的行
        for local_idx, row in df.iterrows():
            if samples_collected >= num_samples:
                break
            
            samples_collected += 1
            
            write_line(f"Sample {samples_collected} (File: {os.path.basename(file_path)}, Row: {local_idx}):")
            write_line("-" * 80)
            
            # 获取文本
            if 'text' in row:
                text = row['text']
                write_line(f"Text length: {len(text)} characters")
                
                # Tokenize
                try:
                    tokens = tokenizer.encode(text)
                    write_line(f"Token count: {len(tokens)}")
                    write_line(f"Token IDs (first 50): {tokens[:50]}")
                except Exception as e:
                    write_line(f"Error tokenizing: {e}")
                    tokens = []
                
                write_line("")
                
                # 打印额外的元数据
                if 'doc_id' in row:
                    write_line(f"Document ID: {row['doc_id']}")
                if 'score' in row:
                    write_line(f"Score: {row['score']}")
                if 'source' in row:
                    write_line(f"Source: {row['source']}")
                
                write_line("")
                write_line("Text content:")
                write_line("~" * 80)
                
                # 打印文本（限制长度）
                if len(text) > 2000:
                    write_line(text[:1000])
                    write_line("")
                    write_line(f"... ({len(text) - 2000} characters omitted) ...")
                    write_line("")
                    write_line(text[-1000:])
                else:
                    write_line(text)
                
                write_line("~" * 80)
            else:
                write_line("Error: No 'text' column found in this row")
            
            write_line("")
            write_line("")
            
            print(f"  Processed sample {samples_collected}")
    
    write_line("=" * 80)
    write_line(f"Completed: {samples_collected} samples written")
    write_line("=" * 80)
    
    if f:
        f.close()
        print(f"\nSuccessfully wrote {samples_collected} samples to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024**2):.2f} MB")


if __name__ == "__main__":
    # Qwen2 tokenizer 路径
    tokenizer_path = "/prodcpfs/user/fengmingquan/model/Qwen2-0.5B"
    
    # 输出目录
    output_dir = "/cpfs/user/fengmingquan/nanoGPT/data/red/samples"
    
    # 数据集配置
    datasets = [
        {
            "name": "books_math_en_0527",
            "path": "/prodcpfs/user/fengmingquan/dataset/raw/books_math_en_0527",
            "output_file": os.path.join(output_dir, "books_math_en_0527_first10.txt")
        },
        {
            "name": "books_math_cn_0527",
            "path": "/prodcpfs/user/fengmingquan/dataset/raw/books_math_cn_0527",
            "output_file": os.path.join(output_dir, "books_math_cn_0527_first10.txt")
        }
    ]
    
    # 处理每个数据集
    for dataset in datasets:
        print("\n" + "=" * 80)
        print(f"Processing: {dataset['name']}")
        print("=" * 80)
        
        if not os.path.exists(dataset['path']):
            print(f"Warning: Dataset directory not found: {dataset['path']}")
            continue
        
        sample_first_n_data(
            dataset_dir=dataset['path'],
            tokenizer_path=tokenizer_path,
            num_samples=10,
            output_file=dataset['output_file']
        )
        
        print("\n")
    
    print("=" * 80)
    print("All datasets processed!")
    print(f"Output files saved in: {output_dir}")
    print("=" * 80)
