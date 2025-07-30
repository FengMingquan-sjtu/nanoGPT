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
import ast

import wandb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from model import GPTConfig, GPT, remove_prefix_from_state_dict
from model_kinet import KINetGPT


os.environ["WANDB_API_KEY"] = "b7f26328382adc825eb193aac3f30f07e7da99c1" 

class QADataset(Dataset):
    """Generic text dataset for different benchmark formats"""
    
    def __init__(self, dataset_name="gsm8k", block_size=1024, tokenizer_name="gpt2"):
        self.block_size = block_size


        # load dataset
        num_proc_load_dataset = 8

        if tokenizer_name.startswith("gpt2"):
            enc = tiktoken.get_encoding("gpt2")
            self.ignore_index = -1
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.ignore_index = -100
        
        if dataset_name == "gsm8k":
            # GSM8K dataset
            dataset = load_dataset("openai/gsm8k", "main", num_proc=num_proc_load_dataset)
            testset = dataset["test"]
            q_name = "question"
            a_name = "answer"
        elif dataset_name == "math":
            # Math dataset
            dataset = load_dataset("json", data_files="/cpfs/user/fengmingquan/math-evaluation-harness/data/math/test.jsonl", num_proc=num_proc_load_dataset)
            testset = dataset["train"]
            q_name = "problem"
            a_name = "solution"

        # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
        def process(example):
            if tokenizer_name.startswith("gpt2"):
                q_ids = enc.encode_ordinary(example[q_name]) # encode_ordinary ignores any special tokens
                a_ids = enc.encode_ordinary(example[a_name]) # encode_ordinary ignores any special tokens
                q_ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
                a_ids.append(enc.eot_token) # add the end of text token, e.g
            else:
                q_ids = tokenizer.encode(example[q_name], add_special_tokens=False)
                a_ids = tokenizer.encode(example[a_name], add_special_tokens=False)
                q_ids.append(tokenizer.eos_token_id)  # add the end of text token
                a_ids.append(tokenizer.eos_token_id)  # add the end of text token

            out = {'q_ids': q_ids, 'a_ids': a_ids, 'len': len(q_ids) + len(a_ids)}
            return out

        # tokenize the dataset
        self.testset_token = testset.map(
            process,
            remove_columns=[q_name, a_name],
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
        y = F.pad(torch_a_ids, (len(torch_q_ids)-1, 0), value=self.ignore_index) # shape: (q_seq_len + a_seq_len - 1,).  Pad with -1 (ignore index) for the question part
        # Ensure x and y are of the same length block size
        if len(x) < self.block_size:
            padding_length = self.block_size - len(x)
            x = F.pad(x, (0, padding_length), value=x[-1])  # right pad x with last token (eos)
            y = F.pad(y, (0, padding_length+1), value=self.ignore_index)  # right pad y with -1 (ignore index)
        elif len(x) > self.block_size:
            x = x[-self.block_size:]
            y = y[-self.block_size:]
        
        return x, y

def load_model(model_path, ckpt_step, device='cuda', model_name='gpt2'):
    """Load trained model from checkpoint"""
    
    if model_name.startswith("gpt2"):
        if ckpt_step>0:
            model_fname = os.path.join(model_path, f'ckpt-{ckpt_step}.pt')
            print(f"Loading model from {model_fname}")
            checkpoint = torch.load(model_fname, map_location=device)
            model_args = checkpoint['model_args']
            
            # Create model
            gptconf = GPTConfig(**model_args)
            if 'kinet' in model_path:
                print(f"Loading KINetGPT model")
                model = KINetGPT(gptconf)
            else:
                print(f"Loading GPT model")
                model = GPT(gptconf)
            
            # Load state dict
            state_dict = checkpoint['model']
            model.load_ckp_state_dict(state_dict)
            print(f"Loaded checkpoint")
            
        else:
            # Try loading as pretrained GPT-2

            model = GPT.from_pretrained(model_name, dict(dropout=0.0))
            print(f"Warning: Loaded pretrained GPT-2 model {model_name}")

    else:
        # Load from HuggingFace model hub
        print(f"Loading model {model_name} from HuggingFace")
        if ckpt_step > 0:
            model_fname = os.path.join(model_path, f'ckpt-{ckpt_step}.pt')
            print(f"Loading model from {model_fname}")
            checkpoint = torch.load(model_fname, map_location=device)
            state_dict = checkpoint['model']
            state_dict = remove_prefix_from_state_dict(state_dict)
            model = AutoModelForCausalLM.from_pretrained(model_name, state_dict=state_dict)
            print(f"Loaded checkpoint")
        else:
            print(f"Loading pretrained model {model_name}")
            model = AutoModelForCausalLM.from_pretrained(model_name)
    
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def evaluate_perplexity(model, dataloader, device='cuda', max_batches=None, model_name='gpt2'):
    """Evaluate perplexity on a dataset"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    ignore_index = dataloader.dataset.ignore_index
    
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
            with torch.no_grad():
                if model_name.startswith('gpt2'):
                    logits, _ = model(x, y)
                else:
                    logits = model(x).logits
                # Calculate loss for each token
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    y.view(-1), 
                    ignore_index=ignore_index, 
                    reduction='sum'
                )
        
        # Count valid tokens (excluding ignore_index)
        valid_tokens = (y != ignore_index).sum().item()

        total_loss += loss.item()
        total_tokens += valid_tokens
        
        # Update progress bar
        #if total_tokens > 0:
        #    current_ppl = math.exp(total_loss / total_tokens)
        #    #pbar.set_postfix({'PPL': f'{current_ppl:.2f}'})
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss, total_tokens

def main():
    parser = argparse.ArgumentParser(description='Evaluate model perplexity on benchmark datasets')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to model checkpoint or pretrained model name')
    parser.add_argument('--model_name', type=str, default='gpt2',
                       help='Model architecture to use, choose from: gpt2, Qwen/Qwen2-1.5B, etc.')
    parser.add_argument('--dataset_name', type=str, default='gsm8k',
                       help='dataset name to use, choose from: gsm8k, math')
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
    
    if not os.path.basename(args.model_path).startswith("20"):
        # Find the latest folder starting with '20' (e.g. 2025-07-28_22-41-45)
        folders = [f for f in os.listdir(args.model_path) if os.path.isdir(os.path.join(args.model_path, f)) and f.startswith('20')]
        args.model_path = os.path.join(args.model_path, sorted(folders)[-1])
        print(f"Auto-detected latest folder: {args.model_path}")
        
    if args.model_name == "auto":
        logfile = os.path.join(args.model_path, "out.log")
        with open(logfile, 'r') as f:
            for line in f:
                if "configs are: " in line:
                    configs = line.split("configs are: ")[-1].strip()
                    configs = ast.literal_eval(configs)
                    args.model_name = configs.get('init_from', 'gpt2')
                    break
        print(f"Auto-detected model name: {args.model_name}")
    if args.wandb_id == "auto":
        logfile = os.path.join(args.model_path, "out.log")
        with open(logfile, 'r') as f:
            for line in f:
                if "wandb:" in line and "/runs/" in line:
                    args.wandb_id = line.split("/runs/")[-1].strip()
                    break
        print(f"Auto-detected WandB ID: {args.wandb_id}")
    

    # Load dataset
    print(f"Loading dataset {args.dataset_name}")
    dataset = QADataset(dataset_name=args.dataset_name, block_size=args.block_size, tokenizer_name=args.model_name)


    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True if 'cuda' in args.device else False
    )
    
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    print(f"Total batches: {len(dataloader)}")
    

    # Load model
    model = load_model(args.model_path, args.ckpt_step, args.device, args.model_name)



    # Evaluate perplexity
    print("Starting evaluation...")
    perplexity, avg_loss, total_tokens = evaluate_perplexity(
        model, dataloader, args.device, args.max_batches, args.model_name
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.model_path}")
    print(f"Total tokens evaluated: {total_tokens:,}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print("="*50)
    
    # Save results if requested
    if args.output_file:
        results = {
            'model_path': args.model_path,
            'dataset_name': args.dataset_name,
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'block_size': args.block_size
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")
    
    if args.wandb_id:
        wandb.init(id=args.wandb_id, resume='must', project="owm")
        wandb.define_metric(f"{args.dataset_name}/ppl", step_metric="train_step")
        wandb.log({
            f'{args.dataset_name}/ppl': perplexity,
            'train_step': args.ckpt_step,
        })
        wandb.finish()
        print(f"Results logged to WandB run {args.wandb_id}")

if __name__ == '__main__':

    main()
    # nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python eval_ppl.py --model_path out/cont-gpt2-1.5B-owm-15B/2025-07-02_21-50-52/ --wandb_id ee485d6a --device "cuda:0" --ckpt_step 0 &