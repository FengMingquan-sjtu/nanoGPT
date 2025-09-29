
import os
import argparse
from eval_ppl import load_model, auto_parse_path
from transformers import AutoModelForCausalLM, AutoTokenizer


def from_pt_load_hf_model(model_path, ckpt_step, model_name):
    model = load_model(model_path, ckpt_step, 'cpu', model_name)
    if model_name.startswith('gpt2'):
        hf_state_dict = model.to_gpt2_state_dict(model.state_dict())
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.load_state_dict(hf_state_dict, strict=True)
    return model



def main():
    parser = argparse.ArgumentParser(description='Convert state_dict from PyTorch to Hugging Face format')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to model checkpoint or pretrained model name')
    parser.add_argument('--model_name', type=str, default='gpt2',
                       help='Model architecture to use, choose from: gpt2, Qwen/Qwen2-1.5B, etc.')
    parser.add_argument('--ckpt_step', type=int, default=0,
                          help='Checkpoint step to load (if applicable)')
    parser.add_argument('--wandb_id', type=str, default="auto",
                       help='WandB run id for logging')
    args = parser.parse_args()

    if args.ckpt_step == 0:
        print("warning: ckpt_step is 0, loading the huggingface checkpoint, not converting from pt checkpoint.")
        return 
    
    args = auto_parse_path(args)
    hf_model_path = os.path.join(args.model_path, f"ckpt-{args.ckpt_step}-hf")

    if not os.path.exists(hf_model_path):
        hf_model = from_pt_load_hf_model(args.model_path, args.ckpt_step, args.model_name)
        hf_model.save_pretrained(hf_model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.save_pretrained(hf_model_path)
        print(f"Converted model saved to {hf_model_path}")
    else:
        print(f"Model already exists at {hf_model_path}, skipping conversion.")

if __name__ == "__main__":
    main()