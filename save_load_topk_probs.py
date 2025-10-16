import os
import re
import time
import numpy as np
import torch


class TopkProbsLoader:
    def __init__(self, topk_probs_dir, top_k, enforce_rank_match=True, load_rank=None):
        self.topk_probs_dir = topk_probs_dir
        self.top_k = int(top_k) if top_k is not None else 0
        os.makedirs(self.topk_probs_dir, exist_ok=True)
        # use distributed rank if available to avoid filename collisions
        self.rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        # control loading behavior
        self.enforce_rank_match = bool(enforce_rank_match)
        self.load_rank = int(load_rank) if load_rank is not None else self.rank
        self._counter = 0

        max_step = 0
        max_rank = 0
        for f in os.listdir(self.topk_probs_dir):
            if not (f.endswith('.npz') and f.startswith('topk_')):
                continue
            # filename is  f"topk_step{step:07d}_rank{rank}.npz"
            step = int(f.split('_')[1].split('step')[1].split('.')[0])
            max_step = max(max_step, step)
            rank = int(f.split('_')[2].split('rank')[1].split('.')[0])
            max_rank = max(max_rank, rank)
        self.max_step = max_step
        self.max_rank = max_rank

    @torch.no_grad()
    def save_topk_probs(self, logits, X, Y):
        """
        Save top-k probabilities and indices for each token position.
        Args:
            logits (torch.Tensor): shape (B, T, V)
            X (torch.Tensor): shape (B, T)
            Y (torch.Tensor): shape (B, T)
        Files written:
            npz with fields: probs (B,T,K) float16, indices (B,T,K) int32, meta dict-like arrays
        """
        if self.top_k is None or self.top_k <= 0:
            return
        if logits is None:
            return

        # ensure tensor on same device, compute probs in float32 for stability
        vocab_size = logits.size(-1)
        k = min(self.top_k, int(vocab_size))
        probs = torch.softmax(logits.float(), dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)

        # move to cpu and cast for compact storage
        probs_np = topk_probs.detach().cpu().numpy().astype(np.float16)
        indices_np = topk_indices.detach().cpu().numpy().astype(np.int32)
        X_np = X.detach().cpu().numpy().astype(np.int32)
        Y_np = Y.detach().cpu().numpy().astype(np.int32)
        # unique, rank-safe filename
        fname = f"topk_step{self._counter:07d}_rank{self.rank}.npz"
        fpath = os.path.join(self.topk_probs_dir, fname)

        np.savez_compressed(
            fpath,
            probs=probs_np,
            indices=indices_np,
            X=X_np,
            Y=Y_np,
            top_k=np.array([k], dtype=np.int32),
            shape=np.array(list(topk_probs.shape) + [vocab_size], dtype=np.int32),
        )
        self._counter += 1

    def load_topk_probs_step(self, batch_size, rand_seed):
        """
        Load top-k probability single file for a specific step.
        Args:
            step: int
        Returns:
            dict with keys:
              - probs: torch.FloatTensor (B,T,K)
              - indices: torch.LongTensor (B,T,K)
              - X: torch.LongTensor (B,T)
              - Y: torch.LongTensor (B,T)
              - top_k: int
              - vocab_size: int (from metadata)
        """        
        step = self._counter % (self.max_step + 1)
        rank = self.rank % (self.max_rank + 1)


        fpath = os.path.join(self.topk_probs_dir, f"topk_step{step:07d}_rank{rank}.npz")
        with np.load(fpath) as data:
            probs = torch.from_numpy(data['probs'].astype(np.float32))
            indices = torch.from_numpy(data['indices'].astype(np.int64))
            X = torch.from_numpy(data['X'].astype(np.int64))
            Y = torch.from_numpy(data['Y'].astype(np.int64))
            top_k = int(data['top_k'][0]) if 'top_k' in data else probs.shape[-1]
            vocab_size = int(data['shape'][-1]) if 'shape' in data else -1
            self._counter += 1
            if probs.shape[0] > batch_size:
                # random sample batch_size from probs
                idx = torch.randint(0, probs.shape[0], (batch_size,), generator=torch.Generator().manual_seed(int(rand_seed)))
                probs = probs[idx].contiguous()
                indices = indices[idx].contiguous()
                X = X[idx].contiguous()
                Y = Y[idx].contiguous()
            else:
                raise ValueError(f"Probs shape {probs.shape} is less than batch size {batch_size}")
            return {
                'probs': probs,
                'indices': indices,
                'X': X,
                'Y': Y,
                'top_k': top_k,
                'vocab_size': vocab_size,
                'path': fpath,
            }
        return None
    def load_topk_probs_step_half(self, step):
        """
        Load top-k probability single file for a specific step. half of batch size used.
        Args:
            step: int
        Returns:
            dict with keys:
              - probs: torch.FloatTensor (B,T,K)
              - indices: torch.LongTensor (B,T,K)
              - X: torch.LongTensor (B,T)
              - Y: torch.LongTensor (B,T)
              - top_k: int
              - vocab_size: int (from metadata)
        """
        half_step = step // 2 
        

        fpath = os.path.join(self.topk_probs_dir, f"topk_step{half_step:07d}_rank{self.rank}.npz")
        with np.load(fpath) as data:
            probs = torch.from_numpy(data['probs'].astype(np.float32))
            indices = torch.from_numpy(data['indices'].astype(np.int64))
            X = torch.from_numpy(data['X'].astype(np.int64))
            Y = torch.from_numpy(data['Y'].astype(np.int64))
            top_k = int(data['top_k'][0]) if 'top_k' in data else probs.shape[-1]
            vocab_size = int(data['shape'][-1]) if 'shape' in data else -1
            half_bs = X.shape[0] // 2
            return {
                'probs': probs[:half_bs] if step % 2 == 0 else probs[half_bs:],
                'indices': indices[:half_bs] if step % 2 == 0 else indices[half_bs:],
                'X': X[:half_bs] if step % 2 == 0 else X[half_bs:],
                'Y': Y[:half_bs] if step % 2 == 0 else Y[half_bs:],
                'top_k': top_k,
                'vocab_size': vocab_size,
                'path': fpath,
            }
        return None

    def load_topk_probs(self):
        """
        Load all saved top-k probability files from directory in sorted order.
        Yields:
            dict with keys:
              - probs: torch.FloatTensor (B,T,K)
              - indices: torch.LongTensor (B,T,K)
              - X: torch.LongTensor (B,T)
              - Y: torch.LongTensor (B,T)
              - top_k: int
              - vocab_size: int (from metadata)
        """
        files = []
        for f in os.listdir(self.topk_probs_dir):
            if not (f.endswith('.npz') and f.startswith('topk_')):
                continue
            if self.enforce_rank_match and self.load_rank is not None:
                prefix = f"topk_rank{self.load_rank}_"
                if not f.startswith(prefix):
                    continue
            files.append(os.path.join(self.topk_probs_dir, f))
        def _parse_sort_keys(path):
            fname = os.path.basename(path)
            # support both patterns:
            #  - new: topk_step{step}_rank{rank}_{ts}.npz
            #  - old: topk_rank{rank}_step{step}_{ts}.npz
            step = -1
            ts = 0
            m = re.search(r"step(\d+)", fname)
            if m:
                step = int(m.group(1))
            m2 = re.search(r"_(\d+)\.npz$", fname)
            if m2:
                ts = int(m2.group(1))
            return (step, ts, fname)

        for fpath in sorted(files, key=_parse_sort_keys):
            with np.load(fpath) as data:
                probs = torch.from_numpy(data['probs'].astype(np.float32))
                indices = torch.from_numpy(data['indices'].astype(np.int64))
                X = torch.from_numpy(data['X'].astype(np.int64))
                Y = torch.from_numpy(data['Y'].astype(np.int64))
                top_k = int(data['top_k'][0]) if 'top_k' in data else probs.shape[-1]
                vocab_size = int(data['shape'][-1]) if 'shape' in data else -1
                yield {
                    'probs': probs,
                    'indices': indices,
                    'X': X,
                    'Y': Y,
                    'top_k': top_k,
                    'vocab_size': vocab_size,
                    'path': fpath,
                }