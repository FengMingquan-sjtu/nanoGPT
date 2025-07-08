import os
from contextlib import nullcontext
import argparse
import pickle
import json

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from typing import List, Dict, Tuple, Set
import tiktoken
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from collections import Counter, defaultdict

from eval_ppl import  GSM8KDataset,load_model
from model import GPT, GPTConfig

class TextDataset(data.Dataset):
    def __init__(self, data_dir, split, block_size, length=None):
        """
        Args:
            data_dir: 数据目录路径
            split: 'train' 或 'val'
            block_size: 序列长度
            length: 数据集大小（None表示使用所有可能的序列）
        """
        self.data_dir = data_dir
        self.split = split
        self.block_size = block_size
        
        # 数据文件路径
        if split == 'train':
            self.data_path = os.path.join(data_dir, 'train.bin')
        else:
            self.data_path = os.path.join(data_dir, 'val.bin')
        
        # 获取数据长度
        self.data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        self.data_length = len(self.data)
        
        # 可能的起始位置数量
        max_length = self.data_length - block_size
        self.length = length if length is not None else max_length
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 重新创建 memmap 以避免内存泄漏
        data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        
        # 随机选择起始位置（保持原有的随机性）
        start_idx = torch.randint(0, len(data) - self.block_size, (1,)).item()
        
        x = torch.from_numpy((data[start_idx:start_idx+self.block_size]).astype(np.int64))
        y = torch.from_numpy((data[start_idx+1:start_idx+1+self.block_size]).astype(np.int64))
        
        return x, y
    
class GPTFeatureExtractor:
    """GPT模型深度特征提取器"""
    
    def __init__(self, model: GPT, layers_to_extract: List[int] = None, 
                 exclude_special_tokens: bool = True):
        """
        Args:
            model: 训练好的GPT模型
            layers_to_extract: 要提取特征的层索引列表，默认提取最后3层
            exclude_special_tokens: 是否排除特殊token
        """
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.exclude_special_tokens = exclude_special_tokens
        
        # 获取tokenizer用于识别特殊token
        self.enc = tiktoken.get_encoding("gpt2")
        self.special_tokens = self._get_special_tokens()
        
        # 默认提取最后几层的特征
        if layers_to_extract is None:
            n_layers = len(model.transformer.h)
            layers_to_extract = [n_layers-3, n_layers-2, n_layers-1]  # 最后3层
        
        self.layers_to_extract = layers_to_extract
        self.features = {}  # 存储提取的特征
        self.hooks = []     # 存储hook句柄
        
        # 注册hook来提取中间层特征
        self._register_hooks()
        
    def _get_special_tokens(self) -> Set[int]:
        """获取需要排除的特殊token ID集合"""
        special_tokens = set()
        
        # GPT-2的特殊token
        special_tokens.add(self.enc.eot_token)  # End-of-text token (50256)
        
        # 添加其他可能的特殊token
        # 如果有其他特殊token，可以在这里添加
        try:
            # 一些常见的特殊token（如果存在的话）
            special_tokens.add(self.enc.encode("<|endoftext|>")[0])
        except:
            pass
            
        # 添加padding token（通常是-1，但我们在token级别处理）
        # 注意：-1通常在targets中使用，在input_ids中一般不会出现
        
        print(f"识别到的特殊token IDs: {special_tokens}")
        return special_tokens
        
    def _register_hooks(self):
        """注册前向传播hook来提取特征"""
        def get_features(name):
            def hook(model, input, output):
                # output shape: (batch_size, seq_len, hidden_dim)
                #self.features[name] = output.detach().cpu()
                self.features[name] = output.detach()
            return hook
        
        # 为指定层注册hook
        for layer_idx in self.layers_to_extract:
            layer = self.model.transformer.h[layer_idx]
            hook = layer.register_forward_hook(get_features(f'layer_{layer_idx}'))
            self.hooks.append(hook)
    
    def _is_special_token(self, token_id: int, token_text: str) -> bool:
        """判断是否为特殊token"""
        if not self.exclude_special_tokens:
            return False
            
        # 检查token ID
        if token_id in self.special_tokens:
            return True
            
        # 检查token文本特征
        token_text = token_text.strip()
        
        # 空白token（通常是padding或特殊空格）
        if not token_text or token_text.isspace():
            return True
            
        # 特殊控制字符
        if token_text.startswith('<|') and token_text.endswith('|>'):
            return True
            
        # 其他可能的特殊模式
        if len(token_text) == 1 and ord(token_text) < 32:  # 控制字符
            return True
            
        return False
    
    def extract_features(self, input_ids: torch.Tensor, 
                        return_tokens: bool = True) -> Dict:
        """
        提取输入token的深度特征
        
        Args:
            input_ids: 输入token序列 (batch_size, seq_len)
            return_tokens: 是否返回对应的token信息
            
        Returns:
            包含特征和token信息的字典
        """
        self.features = {}  # 清空之前的特征
        
        with torch.no_grad():
            # 前向传播，触发hook提取特征
            input_ids = input_ids.to(self.device)
            _ = self.model(input_ids)
        
        # 整理提取的特征
        result = {
            'features': {},
            'input_ids': input_ids.cpu(),
            'special_token_mask': None,  # 用于标记特殊token位置
        }
        
        # 合并所有层的特征
        all_features = []
        for layer_idx in self.layers_to_extract:
            layer_features = self.features[f'layer_{layer_idx}']  # (B, T, H)
            result['features'][f'layer_{layer_idx}'] = layer_features
            all_features.append(layer_features)
        
        # 拼接所有层的特征作为最终特征
        if len(all_features) > 1:
            # 沿着最后一个维度拼接: (B, T, H*n_layers)
            result['features']['combined'] = torch.cat(all_features, dim=-1)
        else:
            result['features']['combined'] = all_features[0]
            
        # 添加token信息和特殊token掩码
        if return_tokens:
            result['tokens'] = []
            batch_size, seq_len = input_ids.shape
            special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            
            for batch_idx in range(batch_size):
                batch_tokens = []
                for pos_idx, token_id in enumerate(input_ids[batch_idx]):
                    if token_id.item() != -1:  # 忽略padding token
                        try:
                            token_str = self.enc.decode([token_id.item()])
                            is_special = self._is_special_token(token_id.item(), token_str)
                            
                            # 更新特殊token掩码
                            special_mask[batch_idx, pos_idx] = is_special
                            
                            batch_tokens.append({
                                'id': token_id.item(),
                                'text': token_str,
                                'type': self._get_token_type(token_str),
                                'is_special': is_special,
                                'position': pos_idx
                            })
                        except:
                            # 无法解码的token也视为特殊token
                            special_mask[batch_idx, pos_idx] = True
                            batch_tokens.append({
                                'id': token_id.item(),
                                'text': f'<UNK_{token_id.item()}>',
                                'type': 'unknown',
                                'is_special': True,
                                'position': pos_idx
                            })
                    else:
                        # padding token
                        special_mask[batch_idx, pos_idx] = True
                        batch_tokens.append({
                            'id': -1,
                            'text': '<PAD>',
                            'type': 'padding',
                            'is_special': True,
                            'position': pos_idx
                        })
                        
                result['tokens'].append(batch_tokens)
            
            result['special_token_mask'] = special_mask
        
        return result
    
    def _get_token_type(self, token: str) -> str:
        """简单的token类型分类"""
        token = token.strip()
        if not token:
            return 'whitespace'
        elif token.isdigit():
            return 'number'
        elif token.isalpha():
            return 'word'
        elif token in '.,!?;:':
            return 'punctuation'
        elif token.startswith(' '):
            return 'word_with_space'
        else:
            return 'other'
    
    def cleanup(self):
        """清理hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class TorchStandardScaler:
    """PyTorch版本的StandardScaler"""
    def __init__(self, device='cuda'):
        self.device = device
        self.mean_ = None
        self.std_ = None
        self.fitted = False
    
    def fit(self, X: torch.Tensor):
        """拟合缩放参数"""
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device, dtype=torch.float32)
        else:
            X = X.to(self.device).float()
            
        self.mean_ = torch.mean(X, dim=0)
        self.std_ = torch.std(X, dim=0, unbiased=False)
        # 避免除零
        self.std_ = torch.where(self.std_ == 0, torch.ones_like(self.std_), self.std_)
        self.fitted = True
        return self
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """应用缩放"""
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet.")
        
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device, dtype=torch.float32)
        else:
            X = X.to(self.device).float()
            
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """拟合并转换"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """逆变换"""
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet.")
            
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device, dtype=torch.float32)
        else:
            X = X.to(self.device).float()
            
        return X * self.std_ + self.mean_

class TorchPCA:
    """PyTorch版本的PCA"""
    def __init__(self, n_components=50, device='cuda'):
        self.n_components = n_components
        self.device = device
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None
        self.fitted = False
    
    def fit(self, X: torch.Tensor):
        """拟合PCA"""
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device, dtype=torch.float32)
        else:
            X = X.to(self.device).float()
        
        # 中心化数据
        self.mean_ = torch.mean(X, dim=0)
        X_centered = X - self.mean_
        
        # SVD分解
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        
        # 保存主成分
        self.components_ = Vt[:self.n_components]
        self.explained_variance_ = (S[:self.n_components] ** 2) / (X.shape[0] - 1)
        self.fitted = True
        return self
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """应用PCA变换"""
        if not self.fitted:
            raise ValueError("PCA has not been fitted yet.")
        
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device, dtype=torch.float32)
        else:
            X = X.to(self.device).float()
        
        X_centered = X - self.mean_
        return torch.mm(X_centered, self.components_.T)
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """拟合并变换"""
        return self.fit(X).transform(X)

class TorchKMeans:
    """PyTorch版本的K-Means"""
    def __init__(self, n_clusters=8, max_iters=100, tol=1e-4, device='cuda', random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.device = device
        self.random_state = random_state
        
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.fitted = False
        
        if random_state is not None:
            torch.manual_seed(random_state)
    
    def _init_centroids(self, X: torch.Tensor) -> torch.Tensor:
        """初始化聚类中心"""
        n_samples, n_features = X.shape
        centroids = torch.zeros(self.n_clusters, n_features, device=self.device)
        
        # K-means++初始化
        # 随机选择第一个中心
        centroids[0] = X[torch.randint(n_samples, (1,))]
        
        for i in range(1, self.n_clusters):
            # 计算到最近中心的距离
            distances = torch.cdist(X, centroids[:i])
            min_distances = torch.min(distances, dim=1)[0]
            
            # 基于距离的概率选择下一个中心
            probabilities = min_distances / torch.sum(min_distances)
            cumulative_probs = torch.cumsum(probabilities, dim=0)
            r = torch.rand(1, device=self.device)
            
            # 选择下一个中心
            selected_idx = torch.searchsorted(cumulative_probs, r)
            selected_idx = torch.clamp(selected_idx, 0, n_samples - 1)
            centroids[i] = X[selected_idx]
        
        return centroids
    
    def fit(self, X: torch.Tensor):
        """拟合K-means"""
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device, dtype=torch.float32)
        else:
            X = X.to(self.device).float()
        
        n_samples, n_features = X.shape
        
        # 初始化聚类中心
        centroids = self._init_centroids(X)
        
        prev_inertia = float('inf')
        
        for iteration in range(self.max_iters):
            # 分配样本到最近的聚类中心
            distances = torch.cdist(X, centroids)
            labels = torch.argmin(distances, dim=1)
            
            # 更新聚类中心
            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if torch.sum(mask) > 0:
                    new_centroids[k] = torch.mean(X[mask], dim=0)
                else:
                    new_centroids[k] = centroids[k]
            
            # 计算inertia（样本到聚类中心的距离平方和）
            inertia = torch.sum((X - centroids[labels]) ** 2)
            
            # 检查收敛
            if abs(prev_inertia - inertia) < self.tol:
                break
            
            centroids = new_centroids
            prev_inertia = inertia
        
        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia
        self.fitted = True
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """预测聚类标签"""
        if not self.fitted:
            raise ValueError("KMeans has not been fitted yet.")
        
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device, dtype=torch.float32)
        else:
            X = X.to(self.device).float()
        
        distances = torch.cdist(X, self.cluster_centers_)
        return torch.argmin(distances, dim=1)
    
    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        """拟合并预测"""
        self.fit(X)
        return self.labels_

class TokenClusteringAnalyzer:
    """Token特征聚类分析器 - GPU加速版本"""
    
    def __init__(self, n_clusters: int = 8, random_state: int = 42,
                 exclude_special_tokens: bool = True, device: str = 'cuda'):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.exclude_special_tokens = exclude_special_tokens
        self.device = device
        
        # 初始化torch组件
        self.scaler = TorchStandardScaler(device=device)
        self.pca = TorchPCA(n_components=50, device=device)  
        self.kmeans = TorchKMeans(n_clusters=n_clusters, device=device, random_state=random_state)
        
        # 用于可视化的t-SNE（保留sklearn版本）
        self.tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
        
        # 存储结果
        self.features_2d = None
        self.cluster_labels = None
        self.token_info = None
        self.fitted = False
        
        # 设置随机种子
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        
    def prepare_data(self, extracted_features: Dict) -> Tuple[torch.Tensor, List]:
        """
        准备聚类数据，过滤掉特殊token
        
        Args:
            extracted_features: 从GPT提取的特征字典
            
        Returns:
            (features_tensor, token_info_list) - 已过滤特殊token，返回GPU张量
        """
        # 获取组合特征
        combined_features = extracted_features['features']['combined']  # (B, T, H)
        tokens_list = extracted_features['tokens']
        
        # 展平为 (total_tokens, hidden_dim)
        batch_size, seq_len, hidden_dim = combined_features.shape
        features_flat = combined_features.view(-1, hidden_dim)
        
        # 转换为GPU张量
        if not isinstance(features_flat, torch.Tensor):
            features_flat = torch.tensor(features_flat, device=self.device, dtype=torch.float32)
        else:
            features_flat = features_flat.to(self.device).float()
        
        # 展平token信息
        token_info_flat = []
        valid_indices = []  # 记录非特殊token的索引
        
        global_idx = 0
        for batch_idx, batch_tokens in enumerate(tokens_list):
            for pos_idx, token_info in enumerate(batch_tokens):
                # 检查是否为特殊token
                is_special = token_info.get('is_special', False)
                
                if self.exclude_special_tokens and is_special:
                    # 跳过特殊token
                    global_idx += 1
                    continue
                
                # 添加额外的位置信息
                token_info_with_pos = {
                    **token_info,
                    'batch_idx': batch_idx,
                    'global_idx': global_idx
                }
                
                token_info_flat.append(token_info_with_pos)
                valid_indices.append(global_idx)
                global_idx += 1
        
        # 只保留非特殊token的特征
        if valid_indices:
            valid_indices = torch.tensor(valid_indices, device=self.device)
            features_filtered = features_flat[valid_indices]
        else:
            print("警告: 所有token都被过滤掉了!")
            features_filtered = features_flat
            
        print(f"原始token数量: {len(features_flat)}")
        print(f"过滤后token数量: {len(features_filtered)}")
        if self.exclude_special_tokens:
            filtered_count = len(features_flat) - len(features_filtered)
            print(f"已过滤特殊token数量: {filtered_count}")
        
        return features_filtered, token_info_flat
    
    def fit_cluster(self, features: Union[torch.Tensor, np.ndarray], 
                   token_info: List[Dict], do_tsne=False) -> Dict:
        """
        执行聚类分析
        
        Args:
            features: token特征矩阵 - 已过滤特殊token
            token_info: token信息列表 - 已过滤特殊token
            
        Returns:
            聚类结果字典
        """
        print(f"开始GPU聚类分析，数据形状: {features.shape}")
        
        # 转换为GPU张量
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, device=self.device, dtype=torch.float32)
        else:
            features = features.to(self.device).float()
        
        if len(features) == 0:
            raise ValueError("没有有效的token用于聚类!")
        
        # 1. 特征标准化
        print("执行标准化...")
        features_scaled = self.scaler.fit_transform(features)
        
        # 2. PCA降维（可选，用于加速）
        if features_scaled.shape[1] > 100:
            print("执行PCA预降维...")
            features_scaled = self.pca.fit_transform(features_scaled)
        
        # 3. K-means聚类
        print(f"执行K-means聚类，k={self.n_clusters}")
        cluster_labels = self.kmeans.fit_predict(features_scaled)
        
        # 4. t-SNE降维用于可视化（转换为CPU进行）
        if do_tsne:
            print("执行t-SNE降维...")
            features_cpu = features_scaled.cpu().numpy()
            features_2d = self.tsne.fit_transform(features_cpu)
        else:
            features_2d = None
        
        # 存储结果
        self.features_2d = features_2d
        self.cluster_labels = cluster_labels.cpu().numpy()
        self.token_info = token_info
        self.fitted = True
        
        # 分析聚类结果
        cluster_analysis = self._analyze_clusters(self.cluster_labels, token_info)
        
        return {
            'features_2d': features_2d,
            'cluster_labels': self.cluster_labels,
            'token_info': token_info,
            'cluster_analysis': cluster_analysis,
            'n_clusters': self.n_clusters,
            'excluded_special_tokens': self.exclude_special_tokens
        }
    
    def predict(self, features: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        对新的特征进行聚类预测（用于训练过程中）
        
        Args:
            features: 新的特征矩阵
            
        Returns:
            聚类标签
        """
        if not self.fitted:
            raise ValueError("Analyzer has not been fitted yet. Call fit_cluster first.")
        
        # 转换为GPU张量
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, device=self.device, dtype=torch.float32)
        else:
            features = features.to(self.device).float()
        
        # 应用相同的预处理步骤
        features_scaled = self.scaler.transform(features)
        
        if hasattr(self.pca, 'fitted') and self.pca.fitted:
            features_scaled = self.pca.transform(features_scaled)
        
        # 预测聚类
        cluster_labels = self.kmeans.predict(features_scaled)
        
        return cluster_labels
    
    def save_state(self, save_path: str):
        """
        保存Analyzer的状态
        
        Args:
            save_path: 保存路径（不包含扩展名）
        """
        if not self.fitted:
            print("警告: Analyzer尚未拟合，保存的状态可能不完整")
        
        state = {
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'exclude_special_tokens': self.exclude_special_tokens,
            'device': self.device,
            'fitted': self.fitted,
        }
        
        # 保存scaler状态
        if self.scaler.fitted:
            state['scaler'] = {
                'mean_': self.scaler.mean_.cpu(),
                'std_': self.scaler.std_.cpu(),
                'fitted': True
            }
        else:
            state['scaler'] = {'fitted': False}
        
        # 保存PCA状态
        if self.pca.fitted:
            state['pca'] = {
                'components_': self.pca.components_.cpu(),
                'explained_variance_': self.pca.explained_variance_.cpu(),
                'mean_': self.pca.mean_.cpu(),
                'n_components': self.pca.n_components,
                'fitted': True
            }
        else:
            state['pca'] = {'fitted': False, 'n_components': self.pca.n_components}
        
        # 保存KMeans状态
        if self.kmeans.fitted:
            state['kmeans'] = {
                'cluster_centers_': self.kmeans.cluster_centers_.cpu(),
                'n_clusters': self.kmeans.n_clusters,
                'inertia_': self.kmeans.inertia_.cpu() if self.kmeans.inertia_ is not None else None,
                'fitted': True
            }
        else:
            state['kmeans'] = {'fitted': False, 'n_clusters': self.kmeans.n_clusters}
        
        # 保存主状态文件
        main_path = f"{save_path}.pkl"
        with open(main_path, 'wb') as f:
            pickle.dump(state, f)
        
        # 保存配置文件（人类可读）
        config_path = f"{save_path}_config.json"
        config = {
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'exclude_special_tokens': self.exclude_special_tokens,
            'device': self.device,
            'fitted': self.fitted,
            'scaler_fitted': self.scaler.fitted,
            'pca_fitted': self.pca.fitted,
            'kmeans_fitted': self.kmeans.fitted,
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Analyzer状态已保存到: {main_path}")
        print(f"配置文件已保存到: {config_path}")
    
    @classmethod
    def load_state(cls, load_path: str, device: str = None):
        """
        从保存的状态加载Analyzer
        
        Args:
            load_path: 加载路径（不包含扩展名）
            device: 目标设备，如果为None则使用保存时的设备
            
        Returns:
            加载的TokenClusteringAnalyzer实例
        """
        main_path = f"{load_path}.pkl"
        
        if not os.path.exists(main_path):
            raise FileNotFoundError(f"State file not found: {main_path}")
        
        with open(main_path, 'rb') as f:
            state = pickle.load(f)
        
        # 使用指定设备或保存时的设备
        target_device = device or state['device']
        
        # 创建新实例
        analyzer = cls(
            n_clusters=state['n_clusters'],
            random_state=state['random_state'],
            exclude_special_tokens=state['exclude_special_tokens'],
            device=target_device
        )
        
        # 恢复scaler状态
        if state['scaler']['fitted']:
            analyzer.scaler.mean_ = state['scaler']['mean_'].to(target_device)
            analyzer.scaler.std_ = state['scaler']['std_'].to(target_device)
            analyzer.scaler.fitted = True
        
        # 恢复PCA状态
        if state['pca']['fitted']:
            analyzer.pca.components_ = state['pca']['components_'].to(target_device)
            analyzer.pca.explained_variance_ = state['pca']['explained_variance_'].to(target_device)
            analyzer.pca.mean_ = state['pca']['mean_'].to(target_device)
            analyzer.pca.n_components = state['pca']['n_components']
            analyzer.pca.fitted = True
        
        # 恢复KMeans状态
        if state['kmeans']['fitted']:
            analyzer.kmeans.cluster_centers_ = state['kmeans']['cluster_centers_'].to(target_device)
            analyzer.kmeans.n_clusters = state['kmeans']['n_clusters']
            if state['kmeans']['inertia_'] is not None:
                analyzer.kmeans.inertia_ = state['kmeans']['inertia_'].to(target_device)
            analyzer.kmeans.fitted = True
        
        analyzer.fitted = state['fitted']
        
        print(f"Analyzer状态已从 {main_path} 加载")
        print(f"目标设备: {target_device}")
        
        return analyzer
    
    def _analyze_clusters(self, cluster_labels: np.ndarray, 
                         token_info: List[Dict]) -> Dict:
        """分析聚类结果"""
        analysis = {}
        
        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            cluster_tokens = [token_info[i] for i in range(len(token_info)) if mask[i]]
            
            if not cluster_tokens:
                continue
                
            # 统计token类型分布
            type_counts = Counter([token['type'] for token in cluster_tokens])
            
            # 统计最常见的token文本
            text_counts = Counter([token['text'] for token in cluster_tokens])
            
            # 统计位置分布
            position_stats = [token['position'] for token in cluster_tokens]
            
            # 统计特殊token数量（如果没有过滤的话）
            special_count = sum(1 for token in cluster_tokens 
                              if token.get('is_special', False))
            
            analysis[cluster_id] = {
                'size': len(cluster_tokens),
                'type_distribution': dict(type_counts),
                'top_tokens': text_counts.most_common(20),
                'avg_position': np.mean(position_stats) if position_stats else 0,
                'position_std': np.std(position_stats) if position_stats else 0,
                'special_token_count': special_count
            }
        
        return analysis
    
    # 保留原有的可视化方法...
    def visualize_clusters(self, results: Dict, save_path: str = None, 
                          figsize: Tuple[int, int] = (15, 10)):
        """可视化聚类结果"""
        features_2d = results['features_2d']
        cluster_labels = results['cluster_labels']
        token_info = results['token_info']
        
        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('GPT Token feature cluster analysis', fontsize=16)
        
        # 1. 按聚类标签着色
        ax1 = axes[0, 0]
        scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=cluster_labels, cmap='tab10', alpha=0.6, s=10)
        ax1.set_title('cluster labels')
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter, ax=ax1)
        
        # 2. 按token类型着色
        ax2 = axes[0, 1]
        token_types = [token['type'] for token in token_info]
        unique_types = list(set(token_types))
        type_colors = {t: i for i, t in enumerate(unique_types)}
        colors = [type_colors[t] for t in token_types]
        
        scatter2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1], 
                             c=colors, cmap='Set3', alpha=0.6, s=10)
        ax2.set_title('color by token type')
        ax2.set_xlabel('t-SNE Dimension 1')
        ax2.set_ylabel('t-SNE Dimension 2')
        
        # 3. 按位置着色
        ax3 = axes[0, 2]
        positions = [token['position'] for token in token_info]
        scatter3 = ax3.scatter(features_2d[:, 0], features_2d[:, 1], 
                             c=positions, cmap='viridis', alpha=0.6, s=10)
        ax3.set_title('color by position')
        ax3.set_xlabel('t-SNE Dimension 1')
        ax3.set_ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter3, ax=ax3)
        
        # 4. 聚类大小分布
        ax4 = axes[1, 0]
        cluster_sizes = [results['cluster_analysis'][i]['size'] 
                        for i in range(self.n_clusters) 
                        if i in results['cluster_analysis']]
        cluster_ids = [i for i in range(self.n_clusters) 
                      if i in results['cluster_analysis']]
        ax4.bar(cluster_ids, cluster_sizes)
        ax4.set_title('size of each cluster')
        ax4.set_xlabel('cluster ID')
        ax4.set_ylabel('number of tokens')
        
        # 5. Token类型在聚类中的分布
        ax5 = axes[1, 1]
        if unique_types and len(results['cluster_analysis']) > 0:
            type_cluster_matrix = np.zeros((len(unique_types), len(cluster_ids)))
            
            for i, cluster_id in enumerate(cluster_ids):
                type_dist = results['cluster_analysis'][cluster_id]['type_distribution']
                for j, token_type in enumerate(unique_types):
                    type_cluster_matrix[j, i] = type_dist.get(token_type, 0)
            
            # 归一化
            row_sums = type_cluster_matrix.sum(axis=1, keepdims=True)
            type_cluster_matrix = type_cluster_matrix / (row_sums + 1e-8)
            
            sns.heatmap(type_cluster_matrix, annot=True, fmt='.2f', 
                       xticklabels=[f'C{i}' for i in cluster_ids],
                       yticklabels=unique_types, ax=ax5, cmap='Blues')
            ax5.set_title('Token type distribution in clusters')
        
        # 6. 位置分布
        ax6 = axes[1, 2]
        if len(results['cluster_analysis']) > 0:
            avg_positions = [results['cluster_analysis'][i]['avg_position'] 
                            for i in cluster_ids]
            position_stds = [results['cluster_analysis'][i]['position_std'] 
                            for i in cluster_ids]
            
            ax6.bar(cluster_ids, avg_positions, yerr=position_stds, capsize=5)
            ax6.set_title('Average position in clusters')
            ax6.set_xlabel('Cluster ID')
            ax6.set_ylabel('Average Position')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")
        
        plt.show()
        
        # 打印详细分析
        self._print_cluster_analysis(results['cluster_analysis'], excluded_special=True)
    
    def _print_cluster_analysis(self, cluster_analysis: Dict, excluded_special: bool):
        """打印聚类分析结果"""
        print("\n" + "="*60)
        title = "聚类分析详细结果 (GPU加速)"
        if excluded_special:
            title += " (已排除特殊Token)"
        print(title)
        print("="*60)
        
        for cluster_id, analysis in cluster_analysis.items():
            print(f"\n聚类 {cluster_id}:")
            print(f"  大小: {analysis['size']} tokens")
            print(f"  平均位置: {analysis['avg_position']:.2f} (±{analysis['position_std']:.2f})")
            
            if analysis['special_token_count'] > 0:
                print(f"  特殊token数量: {analysis['special_token_count']}")
            
            print("  Token类型分布:")
            for token_type, count in analysis['type_distribution'].items():
                percentage = count / analysis['size'] * 100
                print(f"    {token_type}: {count} ({percentage:.1f}%)")
            
            print("  最常见的tokens:")
            for token, count in analysis['top_tokens'][:10]:
                print(f"    '{token}': {count}次")

def quick_test():
    """快速测试聚类功能"""
    
    print("快速聚类测试...")
    
    # 1. 加载预训练模型
    print("加载GPT-2模型...")
    model = GPT.from_pretrained("gpt2", dict(dropout=0.0))
    model.eval()
    
    # 2. 准备测试文本
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    
    test_texts = [
        "Machine learning is a subset of artificial intelligence,Python is a popular programming language for data science, Natural language processing helps computers understand human language. Deep learning is a powerful technique for building neural networks, TensorFlow and PyTorch are popular deep learning frameworks, Data preprocessing is crucial for model performance.",
        "scikit-learn is a Python module for machine learning built on top of SciPy and is distributed under the 3-Clause BSD license. The project was started in 2007 by David Cournapeau as a Google Summer of Code project, and since then many volunteers have contributed. See the About us page for a list of core contributors.",
        "The library provides a range of supervised and unsupervised learning algorithms, including classification, regression, clustering, and dimensionality reduction. It also includes tools for model selection, preprocessing, and evaluation, making it a comprehensive library for machine learning tasks."
    ]
    
    # 编码文本
    encoded_texts = []
    for text in test_texts:
        tokens = enc.encode(text)
        encoded_texts.append(torch.tensor(tokens, dtype=torch.long).unsqueeze(0))
    
    # 拼接成批次
    max_len = max(t.shape[1] for t in encoded_texts)
    padded_texts = []
    for t in encoded_texts:
        if t.shape[1] < max_len:
            padding = torch.full((1, max_len - t.shape[1]), enc.eot_token, dtype=torch.long)
            t = torch.cat([t, padding], dim=1)
        padded_texts.append(t)
    
    input_batch = torch.cat(padded_texts, dim=0)  # (4, max_len)
    print(f"输入形状: {input_batch.shape}")
    
    # 3. 提取特征
    print("提取特征...")
    feature_extractor = GPTFeatureExtractor(model, layers_to_extract=[-1])
    extracted = feature_extractor.extract_features(input_batch)
    
    # 4. 聚类分析
    print("执行聚类...")
    analyzer = TokenClusteringAnalyzer(n_clusters=5)
    features_flat, token_info_flat = analyzer.prepare_data(extracted)
    
    print(f"特征矩阵形状: {features_flat.shape}")
    print(f"Token数量: {len(token_info_flat)}")
    
    # 执行聚类
    results = analyzer.fit_cluster(features_flat, token_info_flat, do_tsne=True)
    
    # 5. 可视化
    print("生成可视化...")
    analyzer.visualize_clusters(results, save_path="quick_test_clustering.png")
    
    # 清理
    feature_extractor.cleanup()
    print("测试完成!")

def run_clustering_experiment(args):
    """运行token聚类实验"""
    
    print("="*60)
    print("GPT Token深度特征聚类实验")
    print("="*60)
    
    # 1. 加载模型
    print(f"\n1. 加载模型: {args.model_path}")
    model = load_model(args.model_path, args.ckpt_step, args.device)
    print(f"模型参数量: {model.get_num_params()/1e6:.2f}M")
    
    # 2. 加载数据
    print(f"\n2. 加载数据集")
    #dataset = GSM8KDataset(block_size=args.block_size)
    dataset = TextDataset(data_dir="/cpfs/user/fengmingquan/dataset/processed-gpt2/open-web-math",
                          split="train",
                          block_size=args.block_size,)

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.batch_size
    )
    print(f"数据集大小: {len(dataset)} samples")
    
    # 3. 初始化特征提取器
    print(f"\n3. 初始化特征提取器")
    # 提取最后几层的特征
    n_layers = len(model.transformer.h)
    layers_to_extract = [n_layers-3, n_layers-2, n_layers-1]
    print(f"提取层索引: {layers_to_extract}")
    
    feature_extractor = GPTFeatureExtractor(model, layers_to_extract)
    
    # 4. 提取特征
    print(f"\n4. 提取特征...")
    all_features = []
    all_tokens = []
    
    max_batches = args.max_batches or 10  # 限制批次数量以加快实验
    
    for batch_idx, (input_ids, _) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
            
        print(f"  处理批次 {batch_idx+1}/{max_batches}")
        
        # 提取特征
        extracted = feature_extractor.extract_features(input_ids)
        
        # 收集数据
        features_flat, token_info_flat = TokenClusteringAnalyzer().prepare_data(extracted)
        all_features.append(features_flat)
        all_tokens.extend(token_info_flat)
        
        if batch_idx == 0:
            print(f"  单批特征形状: {features_flat.shape}")
    
    # 合并所有特征
    
    all_features = torch.vstack(all_features)
    print(f"总特征形状: {all_features.shape}")
    print(f"总token数量: {len(all_tokens)}")
    
    # 5. 执行聚类分析
    print(f"\n5. 执行聚类分析 (k={args.n_clusters})")
    analyzer = TokenClusteringAnalyzer(n_clusters=args.n_clusters)
    
    # 为了加速，可以采样部分数据
    if len(all_tokens) > args.max_tokens:
        print(f"采样 {args.max_tokens} 个tokens进行聚类...")
        #indices = np.random.choice(len(all_tokens), args.max_tokens, replace=False)
        indices = torch.randperm(len(all_tokens))[:args.max_tokens]
        sampled_features = all_features[indices]
        sampled_tokens = [all_tokens[i] for i in indices]
    else:
        sampled_features = all_features
        sampled_tokens = all_tokens
    
    # 执行聚类
    results = analyzer.fit_cluster(sampled_features, sampled_tokens, do_tsne=args.do_tsne)
    
    output_dir = args.output_dir or "clustering_results"
    os.makedirs(output_dir, exist_ok=True)
    
    
    # 6. 可视化结果
    if args.do_tsne:
        print(f"\n6. 生成可视化结果")
        viz_path = os.path.join(output_dir, f"token_clustering_k{args.n_clusters}.png")
        analyzer.visualize_clusters(results, save_path=viz_path)
    else:
        print(f"\n6. 打印聚类分析结果")
        analyzer._print_cluster_analysis(results['cluster_analysis'], excluded_special=True)
    
    # 7. 保存详细结果
    print(f"\n7. 保存结果")
    import pickle
    results_path = os.path.join(output_dir, f"clustering_results_k{args.n_clusters}.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"详细结果已保存到: {results_path}")
    
    # 清理
    feature_extractor.cleanup()
    
    print(f"\n实验完成! 结果保存在: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='GPT Token深度特征聚类实验')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--ckpt_step', type=int, default=0,
                       help='检查点步数')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='设备')
    
    # 数据参数
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--block_size', type=int, default=1024,
                       help='序列长度')  
    parser.add_argument('--max_batches', type=int, default=20,
                       help='最大处理批次数')
    parser.add_argument('--max_tokens', type=int, default=10000,
                       help='用于聚类的最大token数量')
    
    # 聚类参数
    parser.add_argument('--n_clusters', type=int, default=8,
                       help='聚类数量')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='clustering_results',
                       help='输出目录')
    
    parser.add_argument('--do_tsne', action='store_true',
                       help='是否执行t-SNE降维可视化(默认不做)')    
    args = parser.parse_args()
    
    run_clustering_experiment(args)

if __name__ == '__main__':
    main() #python token_cluster.py --model_path "gpt2-xl" --n_clusters 16 --batch_size 8 --max_batches 10 --max_tokens 100000
    #quick_test()  # 运行快速测试
