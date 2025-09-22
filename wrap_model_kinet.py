import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.utils.deprecation import deprecate_kwarg

class KINET_DSMC(nn.Module): 
    ''' Kinetic Monte Carlo Simulation in channel dimension
    Reference https://medium.com/swlh/create-your-own-direct-simulation-monte-carlo-with-python-3b9f26fa05ab
    Reference https://github.com/pmocz/dsmc-python/blob/main/dsmc.py
    '''
    def __init__(self,kinet_dim=4, max_channal=256):
        super().__init__()
        self.dt = 1 # time step
        self.L = 1 # length of hypercube [-L, L]^d
        self.v_r_max = 10 # over-estimated maximum relative velocity
        self.coll_coef = 0.5 # collision coefficient controls the number of max collisions
        self.gamma = 1e-3
        self.kinet_dim = kinet_dim    
        self.max_channal = max_channal
    
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
        dtype = v_r.dtype

        # Get the indices of the upper triangle (excluding the diagonal), total of num_pairs = n(n-1)/2 elements
        indices = torch.triu_indices(n, n, offset=1)  # shape: [2, num_pairs]

        # Extract the independent upper-triangle elements from v_r
        v_r_upper = v_r[:, indices[0], indices[1]] # (bs, num_pairs)

        # Project the upper-triangle scalar to the chnl-dimensional space
        # Expand v_r_upper to shape (bs, 1, num_pairs) and multiply element-wise with the random unit vectors
        rand_vec = rand_vec[:bs, :, :]
        proj_upper = v_r_upper.unsqueeze(1) * rand_vec  # shape: (bs, chnl, num_pairs)

        # Initialize the output tensor with shape (bs, chnl, n, n)
        v_r_leaving = torch.zeros(bs, chnl, n, n, device=device, dtype=dtype)

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
        dtype = x.dtype

        if chnl < chnl_old or chnl % chnl_old != 0:
            return pre_x + pre_a

        n_divide = chnl // chnl_old
        n = int(n_old // n_divide)
        rand_vec = 2 * torch.pi * torch.rand((bs, chnl, n*(n-1)//2), dtype=dtype, device=device)

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
            coll_coef = self.coll_coef
        else:
            coll_coef = 0

        collision_mask = mask > (1 -  coll_coef) # (bs, n, n), mask of particles that collide, equivalent to bernoulli(p=v_r/v_max*u_x)
        
        delta_v = torch.zeros((bs, chnl, n, n))

        v_r_leaving = self.project_to_antisymmetric(v_r, chnl, rand_vec)

        delta_v = v_cm + v_r_leaving - v.unsqueeze(-1)

        v = v + torch.sum(delta_v * collision_mask.unsqueeze(1), dim=2)
        collision_mask = collision_mask.to(dtype)

        eye_matrix = torch.eye(n, device=device, dtype=dtype).unsqueeze(0)  # shape: (1, n, n)
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

#Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py#L219
@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen2_decoder_forward_kinet(self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        cache_position,
        position_embeddings,  # necessary, but kept here for BC
        **kwargs,):
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        
        # ----Old skipp connection----
        hidden_states = residual + hidden_states
        
        # ----New kinet connection----
        kinet = KINET_DSMC(max_channal=hidden_states.shape[1])
        hidden_states = kinet.forward(residual, hidden_states, hidden_states)
        # ----Modification end----


        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def warp_qwen2_kinet(model):
    # input model is Qwen2ForCausalLM
    for decoder in model.model.layers:
        decoder.forward = types.MethodType(qwen2_decoder_forward_kinet, decoder)
    return model


def test_qwen2():
    model = AutoModelForCausalLM.from_pretrained("/prodcpfs/user/fengmingquan/model/Qwen2-0.5B", trust_remote_code=True)
    model = warp_qwen2_kinet(model)
    input_ids = torch.randint(0, 1000, (2, 16))
    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    print(outputs.logits.shape)

if __name__ == "__main__":
    test_qwen2()