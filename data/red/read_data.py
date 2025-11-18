#!/usr/bin/env python3
"""
读取 Parquet 格式的数据集并打印前10条数据（解码为自然语言）
"""

import os
import glob
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def read_and_display_parquet_dataset(dataset_dir, tokenizer_path, num_samples=10):
    """
    读取 Parquet 格式的数据集并显示前 num_samples 条数据
    
    Args:
        dataset_dir: 数据集目录路径（包含 .parquet 文件）
        tokenizer_path: tokenizer 路径
        num_samples: 要显示的样本数量
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
    
    # 读取第一个文件看看数据结构
    print(f"\nReading first file: {os.path.basename(parquet_files[0])}")
    table = pq.read_table(parquet_files[0])
    
    print(f"Schema: {table.schema}")
    print(f"Number of rows in first file: {len(table)}")
    print(f"Column names: {table.column_names}")
    
    print(f"\n{'='*80}")
    print(f"Displaying first {num_samples} samples:")
    print(f"{'='*80}\n")
    
    # 转换为 pandas DataFrame 以便处理
    df = table.to_pandas()
    
    samples_shown = 0
    
    for file_idx, parquet_file in enumerate(parquet_files):
        if samples_shown >= num_samples:
            break
        
        # 读取当前文件
        if file_idx > 0:  # 第一个文件已经读过了
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
        
        # 遍历当前文件中的行
        for idx, row in df.iterrows():
            if samples_shown >= num_samples:
                break
            
            samples_shown += 1
            print(f"Sample {samples_shown} (File: {os.path.basename(parquet_file)}, Row: {idx}):")
            
            # 检查数据结构
            if 'tokens' in row:
                tokens = row['tokens']
                print(f"  Token count: {len(tokens)}")
                print(f"  Token IDs (first 50): {tokens[:50]}")
                
                # 解码为文本
                text = tokenizer.decode(tokens, skip_special_tokens=False)
            elif 'input_ids' in row:
                tokens = row['input_ids']
                print(f"  Token count: {len(tokens)}")
                print(f"  Token IDs (first 50): {tokens[:50]}")
                
                # 解码为文本
                text = tokenizer.decode(tokens, skip_special_tokens=False)
            elif 'text' in row:
                # 如果数据集中直接存储的是文本
                text = row['text']
                print(f"  Text length: {len(text)} characters")
                # 可选：tokenize 看看
                tokens = tokenizer.encode(text)
                print(f"  Token count (after encoding): {len(tokens)}")
                print(f"  Token IDs (first 50): {tokens[:50]}")
            else:
                # 打印所有列以便查看数据结构
                print(f"  Available columns: {list(row.keys())}")
                for col in row.keys():
                    val = row[col]
                    if isinstance(val, (list, tuple)):
                        print(f"  {col}: [{type(val).__name__}, length={len(val)}] {val[:5] if len(val) > 5 else val}")
                    else:
                        print(f"  {col}: {type(val).__name__} = {str(val)[:100]}")
                
                # 尝试找到 token 相关的列
                for col in row.keys():
                    if 'token' in col.lower() or 'id' in col.lower():
                        tokens = row[col]
                        if isinstance(tokens, (list, tuple)):
                            text = tokenizer.decode(tokens, skip_special_tokens=False)
                            break
                else:
                    print("  Warning: Could not find token column, skipping decode")
                    print(f"  {'-'*76}\n")
                    continue
            
            # 打印解码后的文本（限制长度以便查看）
            print(f"  Decoded text:")
            print(f"  {'-'*76}")
            if len(text) > 1000:
                print(f"  {text[:500]}")
                print(f"  ... ({len(text) - 1000} characters omitted) ...")
                print(f"  {text[-500:]}")
            else:
                print(f"  {text}")
            print(f"  {'-'*76}\n")


if __name__ == "__main__":
    # 数据集目录路径
    dataset_dir = "/prodcpfs/user/fengmingquan/dataset/raw/books_math_en_0527"
    #dataset_dir = "/prodcpfs/user/fengmingquan/dataset/raw/books_math_cn_0527"
    
    # 检查目录是否存在
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        exit(1)
    
    # Qwen2 tokenizer 路径
    tokenizer_path = "/prodcpfs/user/fengmingquan/model/Qwen2-0.5B"
    
    # 读取并显示数据
    read_and_display_parquet_dataset(
        dataset_dir=dataset_dir,
        tokenizer_path=tokenizer_path,
        num_samples=10
    )
