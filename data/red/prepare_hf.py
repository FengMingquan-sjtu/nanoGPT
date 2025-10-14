#!/usr/bin/env python3
"""
数据处理脚本：
1. 将多个数据集按权重混合
2. 将 tokens 拼接成固定长度的序列 (batch_size * seq_length)
3. 采样指定数量的 tokens 并保存到新的 indexed dataset
"""

import os
import struct
import numpy as np
from typing import List, Tuple
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer




# ============= 文件路径工具函数 =============

def index_file_path(prefix_path):
    return prefix_path + '.idx'

def data_file_path(prefix_path):
    return prefix_path + '.bin'

def code(dtype):
    """获取 dtype 对应的 code"""
    dtypes = {
        1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32,
        5: np.int64, 6: np.float64, 7: np.float32, 8: np.uint16
    }
    for k, v in dtypes.items():
        if v == dtype:
            return k
    raise ValueError(f"Unknown dtype: {dtype}")


# ============= 索引数据集读取类 =============

class MMapIndexedDataset:
    """读取 Megatron-style 的 indexed dataset (.idx + .bin 文件)"""
    
    class Index:
        _HDR_MAGIC = b'MMIDIDX\x00\x00'
        
        def __init__(self, idx_path):
            with open(idx_path, 'rb') as f:
                magic = f.read(9)
                assert magic == self._HDR_MAGIC, f"Invalid index file: {idx_path}"
                
                version = struct.unpack('<Q', f.read(8))[0]
                assert version == 1
                
                dtype_code = struct.unpack('<B', f.read(1))[0]
                dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32,
                         5: np.int64, 6: np.float64, 7: np.float32, 8: np.uint16}
                self.dtype = dtypes[dtype_code]
                
                self.num_samples = struct.unpack('<Q', f.read(8))[0]
                self.num_docs = struct.unpack('<Q', f.read(8))[0]
                
                offset = f.tell()
            
            bin_buffer = np.memmap(idx_path, mode='r', order='C')
            
            self.sizes = np.frombuffer(
                bin_buffer, dtype=np.int32, count=self.num_samples, offset=offset
            )
            
            self.pointers = np.frombuffer(
                bin_buffer, dtype=np.int64, count=self.num_samples,
                offset=offset + self.sizes.nbytes
            )
            
            self.doc_idx = np.frombuffer(
                bin_buffer, dtype=np.int64, count=self.num_docs,
                offset=offset + self.sizes.nbytes + self.pointers.nbytes
            )
    
    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        idx_path = path_prefix + '.idx'
        bin_path = path_prefix + '.bin'
        
        assert os.path.exists(idx_path), f"Index file not found: {idx_path}"
        assert os.path.exists(bin_path), f"Binary file not found: {bin_path}"
        
        self.index = self.Index(idx_path)
        self.bin_buffer = np.memmap(bin_path, mode='r', order='C')
    
    def __len__(self):
        return self.index.num_samples
    
    def __getitem__(self, idx):
        ptr = self.index.pointers[idx]
        size = self.index.sizes[idx]
        tokens = np.frombuffer(
            self.bin_buffer, dtype=self.index.dtype, count=size, offset=ptr
        )
        return tokens.copy()


# ============= 索引数据集构建类 =============

class MMapIndexedDatasetBuilder:
    """构建新的 indexed dataset"""
    
    class IndexWriter:
        _HDR_MAGIC = b'MMIDIDX\x00\x00'
        
        def __init__(self, path, dtype):
            self._file = open(path, 'wb')
            self._dtype = dtype
            
            # 写入 header
            self._file.write(self._HDR_MAGIC)
            self._file.write(struct.pack('<Q', 1))  # version
            self._file.write(struct.pack('<B', code(dtype)))  # dtype code
        
        @staticmethod
        def _get_pointers(sizes, dtype):
            dtype_size = dtype().itemsize
            address = 0
            pointers = []
            for size in sizes:
                pointers.append(address)
                address += size * dtype_size
            return pointers
        
        def write(self, sizes, modes, doc_idx):
            pointers = self._get_pointers(sizes, self._dtype)
            
            # 写入 num_samples 和 num_docs
            self._file.write(struct.pack('<Q', len(sizes)))
            self._file.write(struct.pack('<Q', len(doc_idx)))
            
            # 写入 sizes
            sizes_arr = np.array(sizes, dtype=np.int32)
            self._file.write(sizes_arr.tobytes(order='C'))
            
            # 写入 pointers
            pointers_arr = np.array(pointers, dtype=np.int64)
            self._file.write(pointers_arr.tobytes(order='C'))
            
            # 写入 doc_idx
            doc_idx_arr = np.array(doc_idx, dtype=np.int64)
            self._file.write(doc_idx_arr.tobytes(order='C'))
        
        def close(self):
            self._file.close()
    
    def __init__(self, out_file, dtype=np.int32):
        self._data_file = open(out_file, 'wb')
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]
    
    def add_item(self, tokens):
        """添加一个样本（固定长度的 token 序列）"""
        if isinstance(tokens, list):
            tokens = np.array(tokens, dtype=self._dtype)
        elif not isinstance(tokens, np.ndarray):
            tokens = np.array(tokens, dtype=self._dtype)
        else:
            tokens = tokens.astype(self._dtype)
        
        self._data_file.write(tokens.tobytes(order='C'))
        self._sizes.append(len(tokens))
    
    def end_document(self):
        """标记文档结束"""
        self._doc_idx.append(len(self._sizes))
    
    def finalize(self, index_file):
        """完成写入，生成索引文件"""
        self._data_file.close()
        
        writer = self.IndexWriter(index_file, self._dtype)
        writer.write(self._sizes, None, self._doc_idx)
        writer.close()


# ============= 简单二进制文件构建类 =============

class SimpleBinaryWriter:
    """
    简单的二进制文件写入器，类似 fineweb-edu 的格式
    只保存纯 tokens，不需要 header 和 pointers
    """
    
    def __init__(self, output_file, total_tokens, dtype=np.uint16):
        """
        Args:
            output_file: 输出文件路径 (.bin)
            total_tokens: 总 token 数量（用于预分配空间）
            dtype: 数据类型 (np.uint16, np.uint32 等)
        """
        self.output_file = output_file
        self.dtype = dtype
        self.total_tokens = total_tokens
        
        # 使用 memmap 创建文件，预分配空间
        self.arr = np.memmap(output_file, dtype=dtype, mode='w+', shape=(total_tokens,))
        self.idx = 0
        
        print(f"Initialized SimpleBinaryWriter:")
        print(f"  File: {output_file}")
        print(f"  Dtype: {dtype}")
        print(f"  Total tokens: {total_tokens:,}")
    
    def add_tokens(self, tokens):
        """
        添加 tokens 到文件
        
        Args:
            tokens: numpy array 或 list of token ids
        """
        if isinstance(tokens, list):
            tokens = np.array(tokens, dtype=self.dtype)
        elif not isinstance(tokens, np.ndarray):
            tokens = np.array(tokens, dtype=self.dtype)
        else:
            tokens = tokens.astype(self.dtype)
        
        # 检查是否超出容量
        if self.idx + len(tokens) > self.total_tokens:
            raise ValueError(f"Exceeding total_tokens capacity: {self.idx + len(tokens)} > {self.total_tokens}")
        
        # 写入数据
        self.arr[self.idx : self.idx + len(tokens)] = tokens
        self.idx += len(tokens)
    
    def finalize(self):
        """完成写入，刷新缓冲区"""
        self.arr.flush()
        print(f"\nFinalized SimpleBinaryWriter:")
        print(f"  File: {self.output_file}")
        print(f"  Tokens written: {self.idx:,}")
        print(f"  File size: {os.path.getsize(self.output_file) / (1024**3):.2f} GB")
        print(f"\n# To read the bin file later, use:")
        print(f"# arr = np.memmap('{self.output_file}', dtype=np.{self.dtype.__name__}, mode='r')")


# ============= 数据拼接和采样类 =============

class DataPacker:
    """
    将多个数据集的 tokens 拼接成固定长度的序列
    """
    
    def __init__(
        self,
        dataset_paths: List[str],
        weights: List[float],
        seq_length: int,
        eod_token_id: int = 0,
        seed: int = 42
    ):
        """
        Args:
            dataset_paths: 数据集路径列表（不含 .idx/.bin 后缀）
            weights: 对应的采样权重
            seq_length: 目标序列长度
            eod_token_id: 文档结束符 token ID (用于填充)
            seed: 随机种子
        """
        print("Loading datasets...")
        self.datasets = [MMapIndexedDataset(path) for path in dataset_paths]
        self.weights = np.array(weights, dtype=np.float64)
        self.weights = self.weights / self.weights.sum()
        
        self.seq_length = seq_length
        self.eod_token_id = eod_token_id
        self.rng = np.random.RandomState(seed)
        
        # 统计信息
        self.dataset_sizes = [len(ds) for ds in self.datasets]
        self.total_samples = sum(self.dataset_sizes)
        
        print(f"\nDataset Statistics:")
        print(f"  Total datasets: {len(self.datasets)}")
        print(f"  Sequence length: {seq_length}")
        print(f"  EOD token ID: {eod_token_id}")
        
        for i, (path, size, weight) in enumerate(zip(dataset_paths, self.dataset_sizes, self.weights)):
            print(f"  [{i}] {os.path.basename(path)}")
            print(f"      Samples: {size:,}, Weight: {weight:.6f}")
    
    def pack_and_save(
        self,
        output_path: str,
        total_tokens: int,
        show_progress: bool = True
    ):
        """
        打包数据并保存到新的 indexed dataset
        
        Args:
            output_path: 输出文件路径（不含后缀）
            total_tokens: 要采样的总 token 数（例如 5e9 表示 5B tokens）
            show_progress: 是否显示进度条
        """
        print(f"\nPacking data to {output_path}")
        print(f"Target tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
        
        # 计算需要生成的序列数
        # 每个序列长度为 seq_length + 1 (用于训练时的 next token prediction)
        num_sequences = int(np.ceil(total_tokens / (self.seq_length + 1)))
        actual_tokens = num_sequences * (self.seq_length + 1)
        
        print(f"Will generate {num_sequences:,} sequences")
        print(f"Actual tokens: {actual_tokens:,} ({actual_tokens/1e9:.2f}B)")
        
        # 创建 builder
        builder = MMapIndexedDatasetBuilder(
            data_file_path(output_path),
            dtype=np.int32
        )
        
        # 缓冲区：存储从各个数据集读取的 tokens
        buffer = []
        buffer_size = 0
        
        # 用于统计每个数据集的使用情况
        dataset_usage = [0] * len(self.datasets)
        
        iterator = range(num_sequences)
        if show_progress:
            iterator = tqdm(iterator, desc="Packing sequences")
        
        for seq_idx in iterator:
            # 从缓冲区或数据集中获取足够的 tokens
            while buffer_size < self.seq_length + 1:
                # 按权重选择数据集
                ds_idx = self.rng.choice(len(self.datasets), p=self.weights)
                dataset_usage[ds_idx] += 1
                
                # 随机选择一个样本
                sample_idx = self.rng.randint(0, self.dataset_sizes[ds_idx])
                tokens = self.datasets[ds_idx][sample_idx]
                
                # 添加到缓冲区
                buffer.extend(tokens.tolist())
                buffer_size += len(tokens)
                
                # 添加 EOD token 作为文档分隔符
                buffer.append(self.eod_token_id)
                buffer_size += 1
            
            # 从缓冲区取出一个完整序列
            sequence = buffer[:self.seq_length + 1]
            buffer = buffer[self.seq_length + 1:]
            buffer_size -= (self.seq_length + 1)
            
            # 确保序列长度正确
            if len(sequence) < self.seq_length + 1:
                # 填充到目标长度
                sequence.extend([self.eod_token_id] * (self.seq_length + 1 - len(sequence)))
            
            # 添加到 builder
            builder.add_item(sequence)
            
            # 每 100 个序列标记一次文档结束（可选）
            if (seq_idx + 1) % 100 == 0:
                builder.end_document()
        
        # 标记最后一个文档结束
        builder.end_document()
        
        # 完成写入
        print("\nFinalizing index file...")
        builder.finalize(index_file_path(output_path))
        
        # 打印统计信息
        print("\n" + "=" * 60)
        print("Dataset Usage Statistics:")
        total_usage = sum(dataset_usage)
        for i, (usage, expected_weight) in enumerate(zip(dataset_usage, self.weights)):
            actual_weight = usage / total_usage if total_usage > 0 else 0
            print(f"  Dataset [{i}]: {usage:,} samples ({actual_weight:.6f})")
            print(f"    Expected weight: {expected_weight:.6f}")
            print(f"    Difference: {(actual_weight - expected_weight):.6f}")
        
        print("\n" + "=" * 60)
        print(f"Successfully saved to:")
        print(f"  {data_file_path(output_path)}")
        print(f"  {index_file_path(output_path)}")
        print(f"Total sequences: {num_sequences:,}")
        print(f"Total tokens: {actual_tokens:,} ({actual_tokens/1e9:.2f}B)")
    
    def pack_and_save_simple(
        self,
        output_file: str,
        total_tokens: int,
        dtype=np.uint32,
        show_progress: bool = True
    ):
        """
        打包数据并保存到简单的二进制文件（类似 fineweb-edu 格式）
        只保存纯 tokens，不需要 header 和 pointers
        
        Args:
            output_file: 输出文件路径（.bin 文件）
            total_tokens: 要采样的总 token 数（例如 5e9 表示 5B tokens）
            dtype: 数据类型 (np.uint16, np.uint32 等)
            show_progress: 是否显示进度条
        """
        print(f"\nPacking data to {output_file} (simple format)")
        print(f"Target tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
        
        # 创建 SimpleBinaryWriter
        writer = SimpleBinaryWriter(output_file, total_tokens, dtype=dtype)
        
        # 缓冲区：存储从各个数据集读取的 tokens
        buffer = []
        buffer_size = 0
        
        # 用于统计每个数据集的使用情况
        dataset_usage = [0] * len(self.datasets)
        
        # 总共写入的 token 数
        tokens_written = 0
        
        # 使用 tqdm 进度条
        if show_progress:
            pbar = tqdm(total=total_tokens, desc="Writing tokens", unit="tokens")
        
        while tokens_written < total_tokens:
            # 从缓冲区或数据集中获取 tokens
            if buffer_size == 0:
                # 按权重选择数据集
                ds_idx = self.rng.choice(len(self.datasets), p=self.weights)
                dataset_usage[ds_idx] += 1
                
                # 随机选择一个样本
                sample_idx = self.rng.randint(0, self.dataset_sizes[ds_idx])
                tokens = self.datasets[ds_idx][sample_idx]
                
                # 添加到缓冲区
                buffer.extend(tokens.tolist())
                buffer_size += len(tokens)
                
                # 添加 EOD token 作为文档分隔符
                buffer.append(self.eod_token_id)
                buffer_size += 1
            
            # 计算这次要写入多少 tokens
            tokens_to_write = min(buffer_size, total_tokens - tokens_written)
            
            # 从缓冲区取出 tokens
            batch = buffer[:tokens_to_write]
            buffer = buffer[tokens_to_write:]
            buffer_size -= tokens_to_write
            
            # 写入文件
            writer.add_tokens(batch)
            tokens_written += tokens_to_write
            
            # 更新进度条
            if show_progress:
                pbar.update(tokens_to_write)
        
        if show_progress:
            pbar.close()
        
        # 完成写入
        print("\nFinalizing file...")
        writer.finalize()
        
        # 打印统计信息
        print("\n" + "=" * 60)
        print("Dataset Usage Statistics:")
        total_usage = sum(dataset_usage)
        for i, (usage, expected_weight) in enumerate(zip(dataset_usage, self.weights)):
            actual_weight = usage / total_usage if total_usage > 0 else 0
            print(f"  Dataset [{i}]: {usage:,} samples ({actual_weight:.6f})")
            print(f"    Expected weight: {expected_weight:.6f}")
            print(f"    Difference: {(actual_weight - expected_weight):.6f}")
        
        print("\n" + "=" * 60)
        print(f"Successfully saved to: {output_file}")
        print(f"Total tokens: {tokens_written:,} ({tokens_written/1e9:.2f}B)")


# ============= 批处理工具 =============

class BatchReader:
    """
    从打包好的数据集中读取批次数据
    """
    
    def __init__(self, dataset_path: str):
        self.dataset = MMapIndexedDataset(dataset_path)
        print(f"Loaded dataset: {dataset_path}")
        print(f"  Total sequences: {len(self.dataset):,}")
    
    def get_batch(self, batch_size: int, batch_idx: int):
        """
        获取一个批次的数据
        
        Args:
            batch_size: 批次大小
            batch_idx: 批次索引（从 0 开始）
        
        Returns:
            numpy array of shape (batch_size, seq_length + 1)
        """
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(self.dataset))
        
        batch = []
        for i in range(start_idx, end_idx):
            tokens = self.dataset[i]
            batch.append(tokens)
        
        # 转换为 numpy array
        batch = np.array(batch, dtype=np.int32)
        return batch
    
    def iter_batches(self, batch_size: int, shuffle: bool = False, seed: int = 42):
        """
        迭代所有批次
        
        Args:
            batch_size: 批次大小
            shuffle: 是否打乱顺序
            seed: 随机种子（当 shuffle=True 时使用）
        
        Yields:
            numpy arrays of shape (batch_size, seq_length + 1)
        """
        num_samples = len(self.dataset)
        indices = np.arange(num_samples)
        
        if shuffle:
            rng = np.random.RandomState(seed)
            rng.shuffle(indices)
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            batch_indices = indices[start_idx:end_idx]
            batch = [self.dataset[i] for i in batch_indices]
            
            yield np.array(batch, dtype=np.int32)


# ============= 配置加载工具 =============

def load_config_from_yaml(yaml_path: str) -> Tuple[List[str], List[float]]:
    """从 YAML 配置文件中加载数据集路径和权重"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    #train_data = config['cybertron']['train_data_path']
    train_data = config['cybertron']['valid_data_path']
    
    weights = []
    paths = []
    for i in range(0, len(train_data), 2):
        weights.append(float(train_data[i]))
        paths.append(train_data[i + 1])
    
    return paths, weights


# ============= 主函数 =============

def main(seed = 47):
    """
    示例用法
    """
    
    # ===== 配置参数 =====
    yaml_path = "/cpfs/user/fengmingquan/cybertron_202510/configs/data_pretrain_v3_average_pai.yaml"
    
    
    seq_length = 4095  # 序列长度（NOTE:不含最后一个用于预测的 token, 所以是4096-1）
    #total_tokens = int(1e9)  # 1B tokens
    total_tokens = int(5e4)  # 50k tokens
    tokenizer_path = "/prodcpfs/user/fengmingquan/model/Qwen2-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    eod_token_id = tokenizer.eos_token_id
    data_type = np.uint32
    
    #output_path = f"/prodcpfs/user/fengmingquan/dataset/processed-qwen2/red-{total_tokens//1e9}bt-seed{seed}"  # 输出路径（不含后缀）
    output_path = f"/prodcpfs/user/fengmingquan/dataset/processed-qwen2/red-1.0bt-seed{seed}"  # 输出路径（不含后缀）
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # ===== 从 YAML 加载配置 =====
    print("Loading configuration from YAML...")
    paths, weights = load_config_from_yaml(yaml_path)
    
    print(f"\nFound {len(paths)} datasets in config")
    
    # 可选：只使用前 N 个数据集进行测试
    # paths = paths[:3]
    # weights = weights[:3]
    
    # 过滤出存在的数据集
    valid_paths = []
    valid_weights = []
    for p, w in zip(paths, weights):
        if os.path.exists(p + '.idx') and os.path.exists(p + '.bin'):
            valid_paths.append(p)
            valid_weights.append(w)
        else:
            print(f"Warning: Dataset not found, skipping: {p}")
    
    if not valid_paths:
        print("Error: No valid datasets found!")
        return
    
    print(f"\nUsing {len(valid_paths)} valid datasets")
    
    # ===== 打包数据 =====
    packer = DataPacker(
        dataset_paths=valid_paths,
        weights=valid_weights,
        seq_length=seq_length,
        eod_token_id=eod_token_id,
        seed=seed
    )
    
    # ===== 方式一：保存为 indexed dataset（包含 .idx 和 .bin） =====
    # packer.pack_and_save(
    #     output_path=output_path,
    #     total_tokens=total_tokens,
    #     show_progress=True
    # )
    
    # ===== 方式二：保存为简单二进制文件（只有 .bin，类似 fineweb-edu 格式） =====
    output_file = os.path.join(output_path, f"val.bin")
    packer.pack_and_save_simple(
        output_file=output_file,
        total_tokens=total_tokens,
        dtype=data_type,  # 根据 vocab size 选择 uint16, uint32 等
        show_progress=True
    )
    
    # ===== 测试读取简单格式 =====
    print("\n" + "=" * 60)
    print("Testing simple format reading...")
    arr = np.memmap(output_file, dtype=data_type, mode='r')
    print(f"Total tokens in file: {len(arr):,}")
    print(f"First 20 tokens: {arr[:20]}")
    print(f"Last 20 tokens: {arr[-20:]}")

    # decode
    print(f"Decoded first 20 tokens: {tokenizer.decode(arr[:20])}")
    print(f"Decoded last 20 tokens: {tokenizer.decode(arr[-20:])}")
    
    # ===== 测试读取 indexed dataset（如果使用方式一） =====
    # 如果使用 pack_and_save 保存的数据，可以这样读取：
    """
    print("\n" + "=" * 60)
    print("Testing batch reading...")
    
    reader = BatchReader(output_path)
    
    # 读取第一个批次
    batch = reader.get_batch(batch_size=4, batch_idx=0)
    print(f"\nFirst batch shape: {batch.shape}")
    print(f"First sample (first 20 tokens): {batch[0, :20]}")
    
    # 测试迭代
    print("\nTesting iteration (first 3 batches)...")
    for i, batch in enumerate(reader.iter_batches(batch_size=8, shuffle=False)):
        print(f"  Batch {i}: shape={batch.shape}")
        if i >= 2:
            break
    """
    
    print("\nDone!")


if __name__ == "__main__":
    for seed in [42,43,44,45,46,47]:
        main(seed)
    # nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python data/red/prepare_hf.py > log/prepare_hf_5.log 2>&1 &