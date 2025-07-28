"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import random
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

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

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
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

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class KINET_DSMC(nn.Module): # by fyf, 改变了发生碰撞的参考系, 将残差链接的输入由v改为a, 将碰撞过程视作n个chl维粒子的碰撞
    ''' Kinetic Monte Carlo Simulation in channel dimension
    Reference https://medium.com/swlh/create-your-own-direct-simulation-monte-carlo-with-python-3b9f26fa05ab
    Reference https://github.com/pmocz/dsmc-python/blob/main/dsmc.py
    '''
    def __init__(self):
        super().__init__()
        self.dt = 1 # time step
        self.L = 1 # length of hypercube [-L, L]^d
        self.v_r_max = 10 # over-estimated maximum relative velocity
        self.coll_coef = 0.5 # collision coefficient controls the number of max collisions
        self.gamma = 1e-3
        self.kinet_dim = 4    
        self.max_channal = 64
    
    def project_to_antisymmetric(self, v_r, chnl, rand_vec):
        """
        Projects the independent elements (upper triangle) of the symmetric matrix v_r (bs, n, n)
        using a random unit vector from a chnl-dimensional spherical distribution, and constructs
        an antisymmetric matrix v_r_leaving (bs, chnl, n, n) with:
        - Diagonal elements set to 0;
        - For i < j, v_r_leaving[:, :, i, j] = v_r[:, i, j] * (random unit vector)
        - For i > j, v_r_leaving[:, :, i, j] = -v_r_leaving[:, :, j, i]
        """
        bs, n, _ = v_r.shape
        device = v_r.device

        # Get the indices of the upper triangle (excluding the diagonal), total of num_pairs = n(n-1)/2 elements
        indices = torch.triu_indices(n, n, offset=1)  # shape: [2, num_pairs]

        # Extract the independent upper-triangle elements from v_r
        v_r_upper = v_r[:, indices[0], indices[1]] # (bs, num_pairs)

        # Project the upper-triangle scalar to the chnl-dimensional space
        # Expand v_r_upper to shape (bs, 1, num_pairs) and multiply element-wise with the random unit vectors
        rand_vec = rand_vec[:bs, :, :]
        proj_upper = v_r_upper.unsqueeze(1) * rand_vec  # shape: (bs, chnl, num_pairs)

        # Initialize the output tensor with shape (bs, chnl, n, n)
        v_r_leaving = torch.zeros(bs, chnl, n, n, device=device)

        # Fill in the upper triangle (i < j) with the projected results
        v_r_leaving[:, :, indices[0], indices[1]] = proj_upper

        # Set the lower triangle (i > j) as the negative of the corresponding upper triangle
        v_r_leaving[:, :, indices[1], indices[0]] = -proj_upper

        return v_r_leaving

    def forward(self, x, v, a):
        '''
        Input:
            x, v, a: position, velocity and acceleration of particles, shape (B, C, X) 
        Output:
            x, v: updated position and velocity of particles, shape (B, C, X) 
        '''
        dim = x.shape[-1]
        collision_dims = torch.randperm(dim)[:self.kinet_dim]
        pre_x = x
        pre_a = a
        x = x[:, :, collision_dims]
        v = v[:, :, collision_dims]
        a = a[:, :, collision_dims]
        bs, chnl_old, n_old = x.shape
        chnl = self.max_channal
        device = x.device 

        if chnl < chnl_old or chnl % chnl_old != 0:
            return pre_x + pre_a

        n_divide = chnl // chnl_old
        n = int(n_old // n_divide)
        rand_vec = 2 * torch.pi * torch.rand((bs, chnl, n*(n-1)//2))
        rand_vec = rand_vec.to(x.device) 

        a = a.view(bs, chnl, n)
        v = v.view(bs, chnl, n)
        x = x.view(bs, chnl, n)

        v = a * self.dt
        
        x_r = x.unsqueeze(-1) - x.unsqueeze(-2) # (bs, chnl, n_particles, n_particles)
        v_r = v.unsqueeze(-1) - v.unsqueeze(-2) # (bs, chnl, n_particles, n_particles)
        v_cm = (v.unsqueeze(-1) + v.unsqueeze(-2)) / 2 # (bs, chnl, n_particles, n_particles)
        x_cm = (x.unsqueeze(-1) + x.unsqueeze(-2)) / 2 # (bs, chnl, n_particles, n_particles)

        x_r = torch.norm(x_r, dim=1)  # (bs, n, n), distance matrix
        v_r = torch.norm(v_r, dim=1)  # (bs, n, n), relative velocity matrix
        u_x = torch.exp(-x_r)  # distance potential, max=1, min=0

        # for each bs, find the maximum relative velocity
        v_r_max, _ = v_r.max(dim=1, keepdim=False)  # (bs, n)
        v_r_max, _ = v_r_max.max(dim=1, keepdim=False)  # (bs)
        v_r_max = v_r_max.view(bs, 1, 1)

        mask = v_r / v_r_max * u_x
        # batchnorm_mask = nn.BatchNorm1d(n).to(device)
        # mask = batchnorm_mask(mask)

        if self.training:
            coll_coef = coll_coef
        else:
            coll_coef = 0

        collision_mask = mask > (1 -  coll_coef) # (bs, n, n), mask of particles that collide, equivalent to bernoulli(p=v_r/v_max*u_x)
        
        delta_v = torch.zeros((bs, chnl, n, n))

        v_r_leaving = self.project_to_antisymmetric(v_r, chnl, rand_vec)

        delta_v = v_cm + v_r_leaving - v.unsqueeze(-1)

        v = v + torch.sum(delta_v * collision_mask.unsqueeze(1), dim=2)
        collision_mask = collision_mask.float()

        eye_matrix = torch.eye(n, device=collision_mask.device).unsqueeze(0)  # shape: (1, n, n)
        eye_matrix = eye_matrix.expand(bs, -1, -1)  # shape: (bs, n, n)
        collision_mask = collision_mask + eye_matrix
        # print(torch.sum(x_cm * collision_mask.unsqueeze(1), dim=2).shape)
        # print(torch.sum(collision_mask, dim=2).unsqueeze(1).shape)
        x = torch.sum(x_cm * collision_mask.unsqueeze(1), dim=2) / torch.sum(collision_mask, dim=2).unsqueeze(1) # 将这一行注释就还原为x不变的版本
        x = x + v * self.dt / 2

        # # 2. Wall collisions
        # # trace the straight-line trajectory to the top wall, bounce it back
        # hit_wall = x.abs() > self.L
        # dt = (self.L - x.abs()[hit_wall]) / v[hit_wall] # time after collision
        # v[hit_wall] = -v[hit_wall]  # reverse velocity
        # dx = torch.zeros_like(x)
        # dx[hit_wall] = 2 * v[hit_wall] * dt # one dt = moving back to wall, another dt = bouncing.
        # x = x + dx

        # add
        pre_x[:, :, collision_dims] = 0
        pre_a[:, :, collision_dims] = 0
        pre_x = pre_x + pre_a

        # collision
        pre_x[:, :, collision_dims] = x.view(bs, int(chnl/n_divide), n_old)
        
        return pre_x


class KINet_Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.kinet_dsmc = KINET_DSMC()
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        attn = self.attn(self.ln_1(x))

        # perform collision
        x = self.kinet_dsmc(x, attn, attn) 

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

class KINetGPT(nn.Module):

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
        # replace the last transformer with the Kinet_block
        self.transformer.h[-1] = KINet_Block(config)

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

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
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
        model = KINetGPT(config)
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
