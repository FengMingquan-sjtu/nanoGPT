"""
Evaluate acc of trained models on benchmark datasets.
"""
import os
import sys
import math
import pickle
import argparse
import json
from contextlib import nullcontext
import ast
import shutil
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import wandb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import tiktoken
import datasets
from datasets import load_dataset, concatenate_datasets # huggingface datasets
import huggingface_hub
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from lm_eval import evaluator


from eval_ppl import load_model, auto_parse_path


os.environ["WANDB_API_KEY"] = "b7f26328382adc825eb193aac3f30f07e7da99c1" 
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#os.environ['NCCL_P2P_DISABLE']='1'
#os.environ['NCCL_IB_DISABLE']='1'
#os.environ['NCCL_DEBUG'] = 'INFO'



def main():
    parser = argparse.ArgumentParser(description='Evaluate model perplexity on benchmark datasets')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to model checkpoint or pretrained model name')
    parser.add_argument('--model_name', type=str, default='gpt2',
                       help='Model architecture to use, choose from: gpt2, Qwen/Qwen2-1.5B, etc.')
    parser.add_argument('--dataset_name', type=str, default='gsm8k',
                       help='dataset name to use, split by comma if multiple')
    parser.add_argument('--ckpt_step', type=int, default=0,
                          help='Checkpoint step to load (if applicable)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--block_size', type=int, default=1024,
                       help='Sequence length')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of samples to evaluate (for quick testing)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='File to save evaluation results')
    parser.add_argument('--wandb_id', type=str, default=None,
                       help='WandB run id for logging')
    parser.add_argument('--n_shot_prompt', type=int, default=0,
                       help='Number of n-shot examples to include in the prompt (default: 0)')
    parser.add_argument('--num_proc_load_dataset', type=int, default=8,
                       help='Number of processes to use for loading dataset (default: 8)')
    parser.add_argument('--backend', type=str, default="hflm",
                       help='Backend to use for evaluation (default: hflm), use vllm for faster evaluation')

    args = parser.parse_args()
    args = auto_parse_path(args)

    hf_model_path = os.path.join(args.model_path, f"ckpt-{args.ckpt_step}-hf")

    if args.backend == "hflm":
        from lm_eval.models.huggingface import HFLM
        print("Starting HFLM evaluation...")
        eval_model=HFLM(pretrained=hf_model_path, tokenizer=hf_model_path),


    elif args.backend == "vllm":
        from lm_eval.models.vllm_causallms import VLLM
        # NOTE: install vllm from source, with pre_compiled 
        eval_model = VLLM(
            pretrained=hf_model_path,  # Path to the model or model name
            tensor_parallel_size=1,  #
            gpu_memory_utilization=0.15, # 0.9 will cause OOM
            trust_remote_code=True,
            dtype="auto",
            max_model_len=args.block_size,
            data_parallel_size=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        )
        
        
    elif args.backend == "sglang":
        from lm_eval.models.sglang_causallms import SGLangLM
        eval_model = SGLangLM(
            pretrained=hf_model_path,  # Path to the model or model name
            dp_size=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        )
    else:
        raise ValueError(f"Unsupported backend: {args.backend}. Choose from 'hflm' or 'vllm'.")
    

    if "gpqa" in args.dataset_name or "synthetic_human" in args.dataset_name:
        # GBQA needs hf login, but network error sometimes.
        key_file = "/cpfs/user/fengmingquan/nanoGPT/hf_key.txt"
        with open(key_file, 'r') as f:
            hf_key = f.read().strip()
        huggingface_hub.login(token=hf_key)


    results = evaluator.simple_evaluate(
        model=eval_model,
        tasks=args.dataset_name.split(','),
        num_fewshot=args.n_shot_prompt,
        batch_size=args.batch_size if args.batch_size > 0 else "auto:4",  # Use 0 for automatic batch size
        limit=args.limit,
        verbosity="WARNING",
        confirm_run_unsafe_code=True,
        write_out=False,
        log_samples=False,
    )
    # only rank 0 will print results
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(results['results'])
        print(results)
        print("="*50)
        
        # Save results if requested
        if args.output_file:
            results = {
                'model_path': args.model_path,
                'dataset_name': args.dataset_name,
                'accuracy': results['results'],
                'block_size': args.block_size
            }
            
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output_file}")
        
        if args.wandb_id and args.wandb_id != "None":
            wandb.init(id=args.wandb_id, resume='must', project="owm")
            for dataset_name, dataset_res in results['results'].items():
                if not dataset_name in args.dataset_name.split(','):
                    continue
                if dataset_name.endswith("_ppl"):
                    dataset_name = dataset_name.replace("_ppl", "")
                for key, value in dataset_res.items():
                    if key == "alias":
                        continue
                    wandb.define_metric(f"{dataset_name}/{key}", step_metric="train_step")
                    wandb.log({
                        f'{dataset_name}/{key}': value,
                        'train_step': args.ckpt_step,
                    })
            wandb.finish()
            print(f"Results logged to WandB run {args.wandb_id}")
        else:
            print("WARNING: No WandB ID provided, no logging.")

if __name__ == '__main__':

    main()
    