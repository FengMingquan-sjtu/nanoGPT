import argparse
import os
import random
from typing import List, Tuple, Optional

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from model import GPT, GPTConfig, token_sort_rho, get_model_name


os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # for debugging


def build_inputs_from_text(tokenizer, text: str, noise_ratio: float, vocab_size: int, seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Tokenize text with tokenizer, then replace a fraction of tokens with random IDs as noise.
    Returns (noisy_ids, noise_mask) where noise_mask[i]=1 if token i was replaced.
    """
    clean_ids = tokenizer.encode(text, add_special_tokens=False)
    rng = random.Random(seed)

    noisy_ids: List[int] = []
    noise_mask: List[int] = []

    for tok in clean_ids:
        if rng.random() < noise_ratio:
            # replace with a random token id in [0, vocab_size)
            noisy_tok = rng.randrange(0, vocab_size)
            noisy_ids.append(noisy_tok)
            noise_mask.append(1)
        else:
            noisy_ids.append(tok)
            noise_mask.append(0)

    return noisy_ids, noise_mask


def decode_tokens_per_piece(tokenizer, token_ids: List[int]) -> List[str]:
    # Decode token IDs into text pieces, handling special tokens.
    pieces = []
    for tok in token_ids:
        if tok == tokenizer.pad_token_id:
            pieces.append("[PAD]")
        elif tok == tokenizer.eos_token_id:
            pieces.append("[EOS]")
        elif tok == tokenizer.unk_token_id:
            pieces.append("[UNK]")
        else:
            pieces.append(tokenizer.decode([tok], skip_special_tokens=True))
    return pieces




def select_tokens(sorted_token_indices: torch.Tensor, t: int, ratio: Optional[float], top_k: Optional[int]) -> torch.Tensor:
    """
    Build boolean mask of selected tokens per row given sorted indices.
    sorted_token_indices: (b, t) from token_sort_rho
    Returns mask (b, t) with True for selected tokens.
    """
    assert (ratio is not None) ^ (top_k is not None), "Specify exactly one of ratio or top_k"
    b = sorted_token_indices.size(0)
    if ratio is not None:
        k = max(0, min(t, int(ratio * t)))
    else:
        k = max(0, min(t, int(top_k)))
    mask = torch.zeros((b, t), dtype=torch.bool, device=sorted_token_indices.device)
    if k > 0:
        topk_indices = sorted_token_indices[:, :k]
        arange_b = torch.arange(b, device=mask.device).unsqueeze(1).expand_as(topk_indices)
        mask[arange_b, topk_indices] = True
    return mask


def render_html(token_pieces: List[str], selected_mask: List[bool], noise_mask: List[bool]) -> str:
    """Render an HTML string where filtered tokens (not selected) are blue, noise is underlined."""
    html_head = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;}",
        ".tok{white-space:pre-wrap;}",
        ".filtered{color:#1f6feb;}",
        ".noise{ text-decoration: underline; text-decoration-style: dashed; }",
        ".legend{margin-bottom:12px;}",
        "</style></head><body>",
        "<div class='legend'>",
        #"<span class='tok'>第一行</span>: 原始文本； ",
        "<span class='filtered'>蓝色</span>: 被选中的token; ",
        #"<span class='noise'>下划线</span>: 注入的噪声token",
        "</div><div>"
    ]
    # Add original text as the first line
    #ori_text = ori_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    #html_parts.append(f"<span class='tok'>{ori_text}</span><br>")
    html_parts = []

    for i, piece in enumerate(token_pieces):
        classes = ["tok"]
        if selected_mask[i]:
            classes.append("filtered")
        if noise_mask[i]:
            classes.append("noise")
        cls = " ".join(classes)
        # Escape HTML special chars in piece minimally
        safe_piece = (piece
                      .replace("&", "&amp;")
                      .replace("<", "&lt;")
                      .replace(">", "&gt;") )
        html_parts.append(f"<span class='{cls}'>{safe_piece}</span>")

    html_tail = ["</div></body></html>"]
    html_head = "".join(html_head)
    html_tail = "".join(html_tail)
    html_parts = "".join(html_parts)

    return html_head, html_parts, html_tail




# usage: /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python visualize_token_filter.py --text_file train_examples.txt --model /prodcpfs/user/fengmingquan/model/Qwen2-0.5B --ref /prodcpfs/user/fengmingquan/model/Qwen2-0.5B-Instruct --ratio 0.6 --noise_ratio 0.0 --output token_filter_vis.html
# /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python visualize_token_filter.py --text_file train_examples.txt --model /prodcpfs/user/fengmingquan/model/Qwen2-1.5B --ref /prodcpfs/user/fengmingquan/model/Qwen2-1.5B-Instruct --ratio 0.6 --noise_ratio 0.0 --output token_filter_vis.html
def main():
    parser = argparse.ArgumentParser(description="Visualize token filtering (rho-1) with noise injection.")
    parser.add_argument("--text_file", type=str, default=None, help="File containing text to visualize (one line)")
    parser.add_argument("--model", type=str, required=True, help="Path to NanoGPT ckpt .pt or HF model id/path for the main model")
    parser.add_argument("--ref", type=str, required=True, help="Reference model path/id (HF or NanoGPT .pt). Default: gpt2")
    parser.add_argument("--ratio", type=float, default=0.8, help="Selection ratio (0-1). Use with --top_k mutually exclusive")
    parser.add_argument("--top_k", type=int, default=None, help="Select fixed top-K tokens (mutually exclusive with --ratio)")
    parser.add_argument("--noise_ratio", type=float, default=0.0, help="Probability of replacing each token with random noise")
    parser.add_argument("--output", type=str, default="token_filter_vis.html", help="Output HTML path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"], help="Override device")
    parser.add_argument("--reverse_select", action="store_true", help="If set, invert selection order in rho-1 (select easiest)")

    args = parser.parse_args()

    # Choose device
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(args.ref, trust_remote_code=True)
    model.to(device)
    ref_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    
    with open(args.text_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    

    html_parts_lst = []
    
    for line in lines:
        text = line.strip()
        if not text:
            continue
    

        noisy_ids, noise_mask = build_inputs_from_text(tokenizer, text, args.noise_ratio, tokenizer.vocab_size, seed=args.seed)

        # Truncate to model block_size if NanoGPT; HF models can often handle longer but keep it simple
        max_len = len(noisy_ids)
        noisy_ids = noisy_ids[:max_len]
        noise_mask = noise_mask[:max_len]

        # Prepare tensors
        input_ids = torch.tensor(noisy_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        # build next-token targets; last target is ignore index (-1) expected by NanoGPT token_sort_rho path
        targets = input_ids.clone()
        targets[:, :-1] = input_ids[:, 1:]
        targets[:, -1] = -1

        # Compute logits for main model
        with torch.no_grad():
            logits =  model(input_ids).logits

        # Run rho-1 sorting
        with torch.no_grad():
            sorted_token_indices, _ = token_sort_rho(
                logits, targets, ref_model, input_ids,
                ratio=args.ratio if args.top_k is None else 1.0,  # ratio irrelevant when using top_k later
                reverse_select=args.reverse_select,
                batch_select=False,
                scale_select=False,
                mask_select=0,
                value_select=False,
                ref_model_2=None,
                smooth_kernel_size=1,
                attn_select=False,
            )

        # Build selection mask
        T = input_ids.size(1)
        sel_mask_tensor = select_tokens(sorted_token_indices, T, ratio=args.ratio if args.top_k is None else None, top_k=args.top_k)
        # don't select the last position (its target is -1), keep mask True there to avoid misleading visualization
        sel_mask_tensor[:, -1] = True
        selected_mask = sel_mask_tensor[0].tolist()

        # Prepare HTML visualization
        token_pieces = decode_tokens_per_piece(tokenizer, noisy_ids)
        html_head, html_parts, html_tail = render_html(token_pieces, selected_mask, [bool(x) for x in noise_mask])
        html_parts_lst.append(html_parts)
    
    
    html = "<hr />".join(html_parts_lst)
    html = html_head + html + html_tail
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)



if __name__ == "__main__":
    main()

