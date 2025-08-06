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
from lm_eval.models.huggingface import HFLM

from model import GPTConfig, GPT, remove_prefix_from_state_dict
from model_kinet import KINetGPT
from eval_ppl import load_model, auto_parse_path


os.environ["WANDB_API_KEY"] = "b7f26328382adc825eb193aac3f30f07e7da99c1" 

def save_to_tmp_hf_folder(model, tokenizer, folder_name):
    ram_disk_path = os.path.join("/dev/shm/", folder_name)
    if not os.path.exists(ram_disk_path):
        os.makedirs(ram_disk_path)
    model.save_pretrained(ram_disk_path)
    tokenizer.save_pretrained(ram_disk_path)
    print(f"Model and tokenizer copied to {ram_disk_path}")
    return ram_disk_path

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
    parser.add_argument('--max_batches', type=int, default=None,
                       help='Maximum number of batches to evaluate (for quick testing)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='File to save evaluation results')
    parser.add_argument('--wandb_id', type=str, default=None,
                       help='WandB run id for logging')
    parser.add_argument('--n_shot_prompt', type=int, default=0,
                       help='Number of n-shot examples to include in the prompt (default: 0)')
    parser.add_argument('--num_proc_load_dataset', type=int, default=8,
                       help='Number of processes to use for loading dataset (default: 8)')
    
    args = parser.parse_args()
    args = auto_parse_path(args)

    # Load model
    model = load_model(args.model_path, args.ckpt_step, args.device, args.model_name)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    #ram_disk_path = save_to_tmp_hf_folder(model, tokenizer, f"{args.model_path}_{args.dataset_name}_{args.ckpt_step}")

    # Evaluate accuracy
    print("Starting evaluation...")
    results = evaluator.simple_evaluate(
        model=HFLM(pretrained=model, tokenizer=tokenizer),
        tasks=args.dataset_name.split(','),
        num_fewshot=args.n_shot_prompt,
        batch_size=args.batch_size,
        limit=int(args.max_batches * args.batch_size) if args.max_batches else None,
        verbosity="WARNING",
    )

    
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(results['results'])
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
    
    if args.wandb_id:
        wandb.init(id=args.wandb_id, resume='must', project="owm", tags=[f"block_size_{args.block_size}", f"vocab_size_{model.config.vocab_size}"])
        for dataset_name, dataset_res in results['results'].items():
            for key, value in dataset_res.items():
                wandb.define_metric(f"{dataset_name}/{key}", step_metric="train_step")
                wandb.log({
                    f'{dataset_name}/{key}': value,
                    'train_step': args.ckpt_step,
                })
        wandb.finish()
        print(f"Results logged to WandB run {args.wandb_id}")
    

    # NOTE: Clean up temporary files
    #if os.path.exists(ram_disk_path):
    #    shutil.rmtree(ram_disk_path)
    

if __name__ == '__main__':

    main()
    