import os
import argparse
import multiprocessing as mp
from typing import List, Tuple, Dict

import numpy as np


def list_common_npz_files(input_dirs: List[str]) -> List[str]:
    """
    List filenames (not paths) that exist in ALL input directories and end with .npz.
    """
    sets = []
    for d in input_dirs:
        files = {f for f in os.listdir(d) if f.endswith('.npz')}
        sets.append(files)
    common = set.intersection(*sets) if sets else set()
    # deterministic order
    return sorted(common)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, mmap_mode='r') as data:
        return {
            'probs': data['probs'].astype(np.float32, copy=False),
            'indices': data['indices'].astype(np.int64, copy=False),
            'X': data['X'].astype(np.int64, copy=False),
            'Y': data['Y'].astype(np.int64, copy=False),
            'top_k': int(data['top_k'][0]) if 'top_k' in data else int(data['probs'].shape[-1]),
            'vocab_size': int(data['shape'][-1]) if 'shape' in data else -1,
            'shape_meta': data['shape'] if 'shape' in data else None,
        }


def ensemble_one(
    fname: str,
    input_dirs: List[str],
    out_dir: str,
    out_top_k: int = None,
) -> Tuple[str, str]:
    """
    Ensemble matching file `fname` across input_dirs and write to out_dir.
    Returns (fname, 'ok'|'skip:reason'|'error:msg').
    """
    try:
        # Load all inputs for this file
        metas = [load_npz(os.path.join(d, fname)) for d in input_dirs]

        # Basic shape checks
        B, T, _ = metas[0]['probs'].shape
        for m in metas[1:]:
            if m['probs'].shape[:2] != (B, T):
                return fname, 'skip:shape_mismatch'

        # Determine output K and vocab size metadata
        input_topks = [m['probs'].shape[-1] for m in metas]
        max_in_k = int(max(input_topks))
        vocab_size = int(metas[0]['vocab_size'])
        desired_k = int(out_top_k) if out_top_k is not None else max_in_k

        # Prepare output arrays
        probs_out = np.zeros((B, T, desired_k), dtype=np.float32)
        indices_out = np.zeros((B, T, desired_k), dtype=np.int64)

        # We'll use indices from the first model as fallback when union < desired_k
        first_indices = metas[0]['indices']

        # For each position, average per token id over union of indices
        for b in range(B):
            for t in range(T):
                sums: Dict[int, float] = {}
                for m in metas:
                    idx_row = m['indices'][b, t]
                    prob_row = m['probs'][b, t]
                    # accumulate
                    for k_idx in range(idx_row.shape[0]):
                        tok = int(idx_row[k_idx])
                        p = float(prob_row[k_idx])
                        sums[tok] = sums.get(tok, 0.0) + p

                # average over number of models
                if sums:
                    num_models = float(len(metas))
                    items = [(tok, val / num_models) for tok, val in sums.items()]
                    # sort by prob desc
                    items.sort(key=lambda x: x[1], reverse=True)
                else:
                    items = []

                # Select up to desired_k
                selected = items[:desired_k]

                # If fewer than desired_k, pad indices using first model's indices (no duplicate)
                if len(selected) < desired_k:
                    existing = {tok for tok, _ in selected}
                    for tok in first_indices[b, t]:
                        tok = int(tok)
                        if tok not in existing:
                            selected.append((tok, 0.0))
                            existing.add(tok)
                            if len(selected) >= desired_k:
                                break

                # Write into output arrays
                if selected:
                    out_indices = [tok for tok, _ in selected]
                    out_probs = np.array([p for _, p in selected], dtype=np.float32)
                    s = float(out_probs.sum())
                    if s > 0.0:
                        out_probs /= s
                else:
                    # degenerate case
                    out_indices = [int(first_indices[b, t, i]) for i in range(desired_k)]
                    out_probs = np.full((desired_k,), 1.0 / max(1, desired_k), dtype=np.float32)

                indices_out[b, t] = np.array(out_indices, dtype=np.int64)
                probs_out[b, t] = out_probs.astype(np.float32, copy=False)

        # Cast and prepare metadata to save compatibly with existing loader
        probs_np = probs_out.astype(np.float16)
        indices_np = indices_out.astype(np.int32)
        X_np = metas[0]['X'].astype(np.int32, copy=False)
        Y_np = metas[0]['Y'].astype(np.int32, copy=False)

        shape_arr = np.array([B, T, desired_k, vocab_size], dtype=np.int32)
        top_k_arr = np.array([desired_k], dtype=np.int32)

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)
        np.savez_compressed(
            out_path,
            probs=probs_np,
            indices=indices_np,
            X=X_np,
            Y=Y_np,
            top_k=top_k_arr,
            shape=shape_arr,
        )
        return fname, 'ok'
    except Exception as e:
        return fname, f'error:{type(e).__name__}:{e}'


def worker(task: Tuple[str, List[str], str, int]):
    fname, input_dirs, out_dir, out_top_k = task
    return ensemble_one(fname, input_dirs, out_dir, out_top_k)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Ensemble top-k probs/indices from multiple folders of npz files.')
    p.add_argument('--inputs', nargs='+', required=True, help='Input folders containing matching .npz files')
    p.add_argument('--out', required=True, help='Output folder to write ensembled .npz files')
    p.add_argument('--processes', type=int, default=max(1, mp.cpu_count() - 1), help='Number of worker processes')
    p.add_argument('--topk', type=int, default=None, help='Output top-k (default: max top-k across inputs)')
    p.add_argument('--limit', type=int, default=None, help='Limit number of files (for quick runs)')
    return p.parse_args()

    # nohup python /cpfs/user/fengmingquan/nanoGPT/ensemble_prob.py --inputs out-prodcpfs/qwen2-0.5B-red-topk-probs-1 out-prodcpfs/qwen2-0.5B-red-topk-probs-2 out-prodcpfs/qwen2-0.5B-red-topk-probs-3 out-prodcpfs/qwen2-0.5B-red-topk-probs-4 out-prodcpfs/qwen2-0.5B-red-topk-probs-5 --out out-prodcpfs/qwen2-0.5B-red-topk-probs-esb-data-5 --topk 50 --processes 80 > log/ensemble_prob_5.log 2>&1 &

def main():
    args = parse_args()
    input_dirs = [os.path.abspath(d) for d in args.inputs]
    out_dir = os.path.abspath(args.out)

    for d in input_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f'Input directory not found: {d}')

    os.makedirs(out_dir, exist_ok=True)

    common_files = list_common_npz_files(input_dirs)
    if args.limit is not None:
        common_files = common_files[: int(args.limit)]

    if not common_files:
        print('No common .npz files found across inputs. Nothing to do.')
        return

    tasks = [(fname, input_dirs, out_dir, args.topk if args.topk is not None else None) for fname in common_files]

    print(f'Ensembling {len(tasks)} files with {args.processes} processes...')
    with mp.Pool(processes=int(args.processes)) as pool:
        for fname, status in pool.imap_unordered(worker, tasks, chunksize=1):
            print(f'[{status}] {fname}')


if __name__ == '__main__':
    main()


