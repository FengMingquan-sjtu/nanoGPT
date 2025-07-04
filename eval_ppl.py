"""
Evaluate perplexity of trained models on benchmark datasets.
"""
import os
import sys
import math
import pickle
import argparse
import json
from contextlib import nullcontext

import wandb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

from model import GPTConfig, GPT


os.environ["WANDB_API_KEY"] = "b7f26328382adc825eb193aac3f30f07e7da99c1" 

class GSM8KDataset(Dataset):
    """Generic text dataset for different benchmark formats"""
    
    def __init__(self, block_size=1024):
        self.block_size = block_size


        # load dataset
        num_proc_load_dataset = 4
        enc = tiktoken.get_encoding("gpt2")
        
        # GSM8K dataset
        dataset = load_dataset("openai/gsm8k", "main", num_proc=num_proc_load_dataset)
        testset = dataset["test"]
        
        # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
        def process(example):
            q_ids = enc.encode_ordinary(example['question']) # encode_ordinary ignores any special tokens
            q_ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
            a_ids = enc.encode_ordinary(example['answer']) # encode_ordinary ignores any special tokens
            a_ids.append(enc.eot_token) # add the end of text token, e.g
            out = {'q_ids': q_ids, 'a_ids': a_ids, 'len': len(q_ids) + len(a_ids)}
            return out

        # tokenize the dataset
        self.testset_token = testset.map(
            process,
            remove_columns=['question', 'answer'],
            desc="tokenizing the splits",
            num_proc=num_proc_load_dataset,
        )
    
    def __len__(self):
        return len(self.testset_token)
    
    def __getitem__(self, idx):
        q_ids = self.testset_token[idx]['q_ids']
        a_ids = self.testset_token[idx]['a_ids']
        torch_q_ids = torch.tensor(q_ids, dtype=torch.int64) #shape: (q_seq_len,)
        torch_a_ids = torch.tensor(a_ids, dtype=torch.int64) #shape: (a_seq_len,)
        x = torch.cat((torch_q_ids, torch_a_ids), dim=0)  # shape: (q_seq_len + a_seq_len,)
        y = torch.cat((torch_q_ids[1:], torch_a_ids), dim=0)  # shape: (q_seq_len + a_seq_len - 1,)
        # Ensure x and y are of the same length block size
        if len(x) < self.block_size:
            padding_length = self.block_size - len(x)
            x = F.pad(x, (0, padding_length), value=x[-1])  # pad x with last token (eos)
            y = F.pad(y, (0, padding_length+1), value=-1)  # pad y with -1 (ignore index)
        elif len(x) > self.block_size:
            x = x[-self.block_size:]
            y = y[-self.block_size:]
        
        return x, y

def load_model(model_path, ckpt_step, device='cuda'):
    """Load trained model from checkpoint"""
    
    
    if ckpt_step>0:
        model_fname = os.path.join(model_path, f'ckpt-{ckpt_step}.pt')
        print(f"Loading model from {model_fname}")
        checkpoint = torch.load(model_fname, map_location=device)
        model_args = checkpoint['model_args']
        
        # Create model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
        # Load state dict
        state_dict = checkpoint['model']
        model.load_ckp_state_dict(state_dict)
        print(f"Loaded checkpoint")
        
    else:
        # Try loading as pretrained GPT-2
        model = GPT.from_pretrained("gpt2-xl", dict(dropout=0.0))
        print(f"Warning: Loaded pretrained GPT-2 model")
    
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def evaluate_perplexity(model, dataloader, device='cuda', max_batches=None):
    """Evaluate perplexity on a dataset"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    # Setup mixed precision context
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    #pbar = tqdm(dataloader, desc="Evaluating PPL")
    
    for batch_idx, (x, y) in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break
            
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

        with ctx:
            logits, _ = model(x, y)
            # Calculate loss for each token
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                y.view(-1), 
                ignore_index=-1, 
                reduction='sum'
            )
        
        # Count valid tokens (excluding ignore_index)
        valid_tokens = (y != -1).sum().item()
        
        total_loss += loss.item()
        total_tokens += valid_tokens
        
        # Update progress bar
        if total_tokens > 0:
            current_ppl = math.exp(total_loss / total_tokens)
            #pbar.set_postfix({'PPL': f'{current_ppl:.2f}'})
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss, total_tokens

def main():
    parser = argparse.ArgumentParser(description='Evaluate model perplexity on benchmark datasets')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to model checkpoint or pretrained model name')
    parser.add_argument('--data_path', type=str, default='openai/gsm8k',
                       help='Path to dataset or dataset name (default: openai/gsm8k)')
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
    
    args = parser.parse_args()
    
    
    
    # Load dataset
    print(f"Loading dataset {args.data_path}")
    dataset = GSM8KDataset(block_size=args.block_size)
    

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if 'cuda' in args.device else False
    )
    
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    print(f"Total batches: {len(dataloader)}")
    

    # Load model
    model = load_model(args.model_path, args.ckpt_step, args.device)
    print(f"Model loaded successfully. Parameters: {model.get_num_params()/1e6:.2f}M")


    # Evaluate perplexity
    print("Starting evaluation...")
    perplexity, avg_loss, total_tokens = evaluate_perplexity(
        model, dataloader, args.device, args.max_batches
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Dataset: {args.data_path}")
    print(f"Model: {args.model_path}")
    print(f"Total tokens evaluated: {total_tokens:,}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print("="*50)
    
    # Save results if requested
    if args.output_file:
        results = {
            'model_path': args.model_path,
            'data_path': args.data_path,
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'block_size': args.block_size
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")
    
    if args.wandb_id:
        wandb.init(id=args.wandb_id, resume='must', project="owm")
        wandb.define_metric("gsm8k/perplexity", step_metric="step")
        wandb.log({
            'gsm8k/perplexity': perplexity,
            'step': args.ckpt_step,
        })
        wandb.finish()
        print(f"Results logged to WandB run {args.wandb_id}")

if __name__ == '__main__':

    main()
    # nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python eval_ppl.py --model_path out/cont-gpt2-1.5B-owm-15B/2025-07-02_21-50-52/ --wandb_id ee485d6a --device "cuda:0" --ckpt_step 0 &