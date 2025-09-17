"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=(attn_mask is None))
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            mask = self.bias if attn_mask is None else attn_mask
            att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, attn_mask=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, attn_mask)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    @classmethod
    def to_gpt2_state_dict(cls, nanogpt_state_dict):
        unwanted_prefix = '_orig_mod.'
        for k,v in list(nanogpt_state_dict.items()):
            if k.startswith(unwanted_prefix):
                nanogpt_state_dict[k[len(unwanted_prefix):]] = nanogpt_state_dict.pop(k)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        sd_hf = dict() # make a copy of the state dict
        for k in nanogpt_state_dict.keys():
            if any(k.endswith(w) for w in transposed):                
                with torch.no_grad():
                    sd_hf[k] = nanogpt_state_dict[k].t().clone()
            else:
                sd_hf[k] = nanogpt_state_dict[k].clone()
        return sd_hf

    def load_ckp_state_dict(self, ckp_state_dict):
        """
        Load a state dict from a checkpoint, which may have been saved with a different config.
        This is useful for resuming training or evaluation from a checkpoint.
        """
        # remove the unwanted prefix if it exists
        ckp_state_dict = remove_prefix_from_state_dict(ckp_state_dict, '_orig_mod.')
        # load the state dict into the model
        self.load_state_dict(ckp_state_dict, strict=False)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    def configure_muon_optimizer(self, weight_decay, learning_rate, betas, device_type):
        hidden_matrix_params = [p for n, p in self.transformer.h.named_parameters() if p.ndim >= 2 and "embed" not in n]
        embed_params = [p for n, p in self.named_parameters() if "embed" in n]
        scalar_params = [p for p in self.parameters() if p.ndim < 2]
        head_params = [self.lm_head.weight]

        # init the optimizer(s)
        # small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
        # discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
        from muon import MuonWithAuxAdam
        adam_groups = [dict(params=head_params, lr=learning_rate), dict(params=embed_params, lr=learning_rate), dict(params=scalar_params, lr=learning_rate)]
        adam_groups = [dict(**g, betas=betas, eps=1e-10, use_muon=False) for g in adam_groups]
        muon_group = dict(params=hidden_matrix_params, lr=learning_rate, momentum=0.95, use_muon=True)
        param_groups = [*adam_groups, muon_group]
        optimizer = MuonWithAuxAdam(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        print(f"Muon optimizer configured with {len(hidden_matrix_params)} hidden matrix parameters, "
                f"{len(embed_params)} embedding parameters, {len(scalar_params)} scalar parameters, "
                f"{len(head_params)} head parameters, and {len(optimizer.param_groups)} total parameter groups.")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

def remove_prefix_from_state_dict(ckp_state_dict, unwanted_prefix = '_orig_mod.'):
    for k,v in list(ckp_state_dict.items()):
        if k.startswith(unwanted_prefix):
            ckp_state_dict[k[len(unwanted_prefix):]] = ckp_state_dict.pop(k)
    return ckp_state_dict

def configure_AdamW_optimizer(model, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")
    return optimizer


def get_model_name(model):
    if hasattr(model, 'module'):
        model = model.module

    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    
    return model.__class__.__name__


def token_sort_rho(logits, targets, ref_model, idx, ratio, reverse_select=False, batch_select=False, scale_select=False, mask_select=0, value_select=False, ref_model_2=None, smooth_kernel_size=1, attn_select=False):
    ignore_idx = -1 # the index of the token to ignore in the targets
    b, t, vocab_size = logits.size()
    #logits shape (b, t, vocab_size), targets shape (b, t), token_loss shape (b*t)
    
    with torch.no_grad():
        token_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=ignore_idx, reduction='none')
        
        
        ref_model.eval() # set the reference model to eval mode
        if get_model_name(ref_model) == 'GPT':
            ref_logits, _ = ref_model(idx)
        elif attn_select:
            output = ref_model(idx, output_attentions=True)
            ref_logits = output.logits
            attn_weights = output.attentions[-5].mean(dim=1)  # average over all heads, shape (b, t, t)
        else:
            ref_logits = ref_model(idx).logits
        ref_token_loss = F.cross_entropy(ref_logits.view(-1, vocab_size), targets.view(-1), ignore_index=ignore_idx, reduction='none')
        diff_token_loss = token_loss - ref_token_loss # shape (b*t,)
        diff_token_loss = diff_token_loss.view(b, -1) # reshape to (b, t)
        sorted_diff_loss = torch.sort(diff_token_loss, descending=not reverse_select, dim=1)
        sorted_token_indices = sorted_diff_loss.indices #shape (b, t)

        if attn_select:
            rho_scores = torch.zeros_like(sorted_token_indices) # shape (b, t)
            # assign rho scores by indices
            ranks = torch.empty_like(rho_scores, dtype=torch.long)
            arange_t = torch.arange(t, device=rho_scores.device).expand(b, t)
            ranks.scatter_(1, sorted_token_indices, arange_t)
            ranks = ranks.float() # convert to float for division
            ranks[ranks >= t * 0.5] = 0.0 # set the lower half to 0
            ranks[ranks < t * 0.5] = 1.0 
            attn_weights = attn_weights.transpose(1, 2) # shape (b, t, t)
            ranked_attn_weights = torch.bmm(attn_weights, ranks.unsqueeze(-1)).squeeze() # shape (b, t)
            #print(f"ranked_attn_weights shape: {ranked_attn_weights.shape}")
            sorted_ranked_attn_weights = torch.sort(ranked_attn_weights, descending=not reverse_select, dim=1)
            #print(f"sorted_ranked_attn_weights shape: {sorted_ranked_attn_weights.values.shape}")
            sorted_ranked_attn_indices = sorted_ranked_attn_weights.indices # shape (b, t). the score of input tokens
            sorted_ranked_attn_indices = sorted_ranked_attn_indices - 1  # shift the indices, so the score is of output tokens
            sorted_ranked_attn_indices = sorted_ranked_attn_indices[sorted_ranked_attn_indices >= 0].reshape(b, -1) # shape (b, t)
            #print(f"sorted_ranked_attn_indices shape: {sorted_ranked_attn_indices.shape}, sorted_token_indices shape: {sorted_token_indices.shape}")
            n_tokens = int(ratio * t * 0.5)+1  # number of tokens to select, half rho-1, half attn
            sorted_token_indices = torch.cat((sorted_token_indices[:,:n_tokens], sorted_ranked_attn_indices[:,:n_tokens]), dim=1) # shape (b, 2*t)
        return sorted_token_indices, diff_token_loss


def get_loss_rho(logits, targets, ref_model=None, idx=None, ratio=0.5, reverse_select=False, batch_select=False, scale_select=False, mask_select=0, value_select=False, ref_model_2=None, smooth_kernel_size=1, attn_select=False):
    """
    Calculate the loss given the logits and targets.
    If ref_model is provided, use it to calculate the reference logits for KL divergence.
    """
    ignore_idx = -1 # the index of the token to ignore in the targets
    if ref_model is not None: #selection tokens by rho-1 algorithm.
        b, t, vocab_size = logits.size()
        #logits shape (b, t, vocab_size), targets shape (b, t), token_loss shape (b*t)
        token_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=ignore_idx, reduction='none')


        
        sorted_token_indices, _ = token_sort_rho(logits, targets, ref_model, idx, ratio, reverse_select, batch_select, scale_select, mask_select, value_select, ref_model_2, smooth_kernel_size, attn_select)
        # gather the average loss for the top-k tokens
        ignore_mask = (targets.view(-1) == ignore_idx) # shape (b*t,)
        loss = token_loss.masked_fill(ignore_mask, 0.0)  #shape (b*t,)

        if batch_select:
            n_ignore = ignore_mask.sum().item() # number of ignored tokens in the batch
            n_tokens = t*b - n_ignore # number of valid tokens in the batch
            n_tokens = int(ratio * n_tokens) # number of tokens to select
            if n_tokens > 0:
                loss = loss.gather(0, sorted_token_indices[:n_tokens]) # shape (b, n_tokens)
                loss = loss.mean() # average over the selected tokens


        else:
            loss = loss.reshape(b, -1) # shape (b, t) 
            n_tokens = int(ratio * t)  # number of tokens to select
            loss = loss.gather(1, sorted_token_indices[:, :n_tokens]) # shape (b, t)
            loss = loss.mean()
    else:
        # standard cross-entropy loss
        # reshape logits and targets to be of shape (b*t, vocab_size) and (b*t,)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignore_idx)
    return loss


def get_token_select_indices(logits, targets, ref_logits, token_keep_ratio):
    with torch.no_grad():
        b, t, vocab_size = logits.size()
        token_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction='none')
        ref_token_loss = F.cross_entropy(ref_logits.view(-1, vocab_size), targets.view(-1), reduction='none')
        diff_token_loss = token_loss - ref_token_loss # shape (b*t,)
        diff_token_loss = diff_token_loss.view(b, -1) # reshape to (b, t)
        sorted_diff_loss = torch.sort(diff_token_loss, descending=True, dim=1)
        sorted_token_indices = sorted_diff_loss.indices #shape (b, t)
        n_tokens = int(token_keep_ratio * t)  # number of tokens to select
        selected_token_indices = sorted_token_indices[:, :n_tokens] # shape (b, n_tokens)
        return selected_token_indices


def get_loss_distill(logits, targets, ref_model, idx, temperature=2.0, distill_ratio=0.9, token_keep_ratio=1.0, distill_top_k=0):
    """ Model Distillation
    Calculate the loss given the logits and targets.
    If ref_model is provided, use it to calculate the reference logits for KL divergence.
    """
    if ref_model is not None: #Distill + (optionally) selection tokens by rho-1 algorithm.
        #logits shape (b, t, vocab_size), targets shape (b, t), token_loss shape (b*t)
        b, t, vocab_size = logits.size()
        with torch.no_grad():
            ref_model.eval()
            ref_logits = ref_model(idx).logits
            soft_targets = F.softmax(ref_logits.view(-1, vocab_size) / temperature, dim=-1)
            if distill_top_k > 0:
                # top-k filtering of the soft targets
                topk = min(distill_top_k, vocab_size)
                v, _ = torch.topk(ref_logits.view(-1, vocab_size), topk)
                indices_to_remove = ref_logits.view(-1, vocab_size) < v[:, [-1]]
                soft_targets = soft_targets.masked_fill(indices_to_remove, 0.0)
                # renormalize
                soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)

            #if distill_top_p > 0:
            #    # top-p filtering of the soft targets
            #    sorted_logits, sorted_indices = torch.sort(ref_logits.view(-1, vocab_size), descending=True, dim=-1)
            #    cumulative_probs = torch.cumsum(F.softmax(sorted_logits / temperature, dim=-1), dim=-1)
            #    sorted_indices_to_remove = cumulative_probs > distill_top_p
            #    # shift the indices to the right to keep also the first token above the threshold
            #    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            #    sorted_indices_to_remove[..., 0] = 0
            #    indices_to_remove = torch.zeros_like(soft_targets, dtype=torch.bool).scatter_(1, sorted_indices, sorted_indices_to_remove)
            #    soft_targets = soft_targets.masked_fill(indices_to_remove, 0.0)
            #    # renormalize
            #    soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)

        # mle loss
        hard_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction='none') #shape (b*t,)

        # distill loss
        soft_prob = F.log_softmax(logits.view(-1, vocab_size) / temperature, dim=-1)
        kl_div = F.kl_div(soft_prob, soft_targets, reduction='none').sum(dim=-1) #shape (b*t,)

        # weighted sum
        loss = hard_loss * (1 - distill_ratio) + kl_div * distill_ratio * (temperature ** 2)

        # optionally select top-k tokens by rho-1 algorithm
        if token_keep_ratio < 1.0:
            selected_token_indices = get_token_select_indices(logits, targets, ref_logits, token_keep_ratio)
            loss = loss.view(b, -1) # shape (b, t)
            loss = loss.gather(1, selected_token_indices) # shape (b, n_tokens)
        # average over the all tokens (or selected tokens if token_keep_ratio < 1.0)
        loss = loss.mean()
    else:
        # standard cross-entropy loss
        # reshape logits and targets to be of shape (b*t, vocab_size) and (b*t,)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return loss