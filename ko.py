'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import random
from matplotlib import pyplot as plt
from functools import partial


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


#_, term_width = os.popen('stty size', 'r').read().split()
#term_width = int(term_width)
term_width = 100

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def apply_gradient_collision(layer, collision_rate=0.1, collision_type="wg", noise_scale=1e-4, thres=0.):
    """Modify gradients to repel similar neurons in the last linear layer.(Soft collision)"""
    if layer.weight.grad is None:
        return

    W = layer.weight.data  # Extract weights
    grad = layer.weight.grad  # Extract gradients

    # Normalize weights for cosine similarity computation
    W_norm = torch.nn.functional.normalize(W, p=2, dim=1)
    w_similarity = torch.mm(W_norm, W_norm.T)  # Cosine similarity matrix

    grad_norm = torch.nn.functional.normalize(grad, p=2, dim=1)
    grad_similarity = torch.mm(grad_norm, grad_norm.T)

    similarity = torch.mul(w_similarity, grad_similarity)
    similarity[similarity < thres] = 0.

    # Compute repulsion force: adjust gradients to push apart similar neurons
    if collision_type == "wg":  # commonly used
        repulsion_force = torch.mm(similarity, grad) * collision_rate
    elif collision_type == "wg_i": # commonly used
        repulsion_force = torch.mm(similarity - torch.eye(similarity.shape[1]).to(similarity.device), grad) * collision_rate
    elif collision_type == "wg_d":
        repulsion_force = torch.triu(torch.mm(similarity - torch.eye(similarity.shape[1]).to(similarity.device), grad) * collision_rate)
    elif collision_type == "wg_t":
        repulsion_force = torch.mm(torch.triu(similarity) - torch.eye(similarity.shape[1]).to(similarity.device), grad) * collision_rate
    elif collision_type == "wg_e":
        repulsion_force = torch.mm(similarity - torch.eye(similarity.shape[1]).to(similarity.device), grad)
        repulsion_force = repulsion_force / (torch.linalg.norm(repulsion_force) + 1e-20) * torch.linalg.norm(grad) * collision_rate
    elif collision_type == "wg_ed":
        repulsion_force = torch.mm(similarity - torch.eye(similarity.shape[1]).to(similarity.device), grad)
        repulsion_force = torch.triu(repulsion_force / (torch.linalg.norm(repulsion_force) + 1e-20) * torch.linalg.norm(grad) * collision_rate)
    # else:
    #     repulsion_force = torch.mm(similarity, grad) * collision_rate
    
    # apply thresholding to make sparsity
    # repulsion_force[torch.mul(similarity, grad_similarity) - torch.eye(similarity.shape[1]).to(similarity.device) < thres] = 0.
        
    layer.weight.grad -= repulsion_force 
    layer.weight.grad += torch.randn_like(grad) * noise_scale
    return torch.max(torch.mul(w_similarity, grad_similarity) - torch.eye(similarity.shape[1]).to(similarity.device))

class KiOptimizer(torch.optim.Optimizer):
    '''Last linear layer optimizer'''
    def __init__(self, base_optimizer, layer, collision_rate=0.1, collision_type="wg", noise_scale=1e-4, thres=0.):
        self.base_optimizer = base_optimizer
        self.before_step_fn = partial(apply_gradient_collision_v1, layer=layer, collision_rate=collision_rate, collision_type=collision_type, noise_scale=noise_scale, thres=thres)
        self.param_groups = self.base_optimizer.param_groups  # 必须有这个
        self.defaults = self.base_optimizer.defaults

    def step(self, *args, **kwargs):
        if self.before_step_fn is not None:
            self.before_step_fn()

        self.base_optimizer.step(*args, **kwargs)
    
    def state_dict(self):
        return self.base_optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.base_optimizer.load_state_dict(state_dict)

    def zero_grad(self, *args, **kwargs):
        self.base_optimizer.zero_grad(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.base_optimizer, name)


def project_to_antisymmetric(v_r, chnl):
    """
    Projects the independent elements (upper triangle) of the symmetric matrix v_r (bs, n, n)
    using a random unit vector from a chnl-dimensional spherical distribution, and constructs
    an antisymmetric matrix v_r_leaving (bs, chnl, n, n) with:
      - Diagonal elements set to 0;
      - For i < j, v_r_leaving[:, :, i, j] = v_r[:, i, j] * (random unit vector)
      - For i > j, v_r_leaving[:, :, i, j] = -v_r_leaving[:, :, j, i]
    """
    n, _ = v_r.shape
    device = v_r.device

    # Get the indices of the upper triangle (excluding the diagonal), total of num_pairs = n(n-1)/2 elements
    indices = torch.triu_indices(n, n, offset=1)  # shape: [2, num_pairs]
    num_pairs = indices.shape[1]

    # Extract the independent upper-triangle elements from v_r
    v_r_upper = v_r[indices[0], indices[1]] # (bs, num_pairs)

    # For each independent element, generate a random unit vector from a chnl-dimensional sphere,
    # Sample from a normal distribution and then normalize
    rand_vec = torch.randn(chnl, num_pairs, device=device)
    rand_vec = rand_vec / torch.norm(rand_vec, dim=1, keepdim=True) # (bs, chnl, num_pairs)

    # Project the upper-triangle scalar to the chnl-dimensional space
    # Expand v_r_upper to shape (bs, 1, num_pairs) and multiply element-wise with the random unit vectors
    # print(v_r_upper.shape, rand_vec.shape, indices.shape, v_r.shape)
    proj_upper = v_r_upper.unsqueeze(0) * rand_vec  # shape: (bs, chnl, num_pairs)

    # Initialize the output tensor with shape (bs, chnl, n, n)
    v_r_leaving = torch.zeros(chnl, n, n, device=device)

    # Fill in the upper triangle (i < j) with the projected results
    v_r_leaving[:, indices[0], indices[1]] = proj_upper

    # Set the lower triangle (i > j) as the negative of the corresponding upper triangle
    v_r_leaving[:, indices[1], indices[0]] = -proj_upper

    return v_r_leaving

def apply_gradient_collision_v1(layer, collision_rate=0.1, collision_type="wg", noise_scale=1, thres=-1):
    '''Hard collision'''
    W = layer.weight.data  # Extract weights
    grad = layer.weight.grad  # Extract gradients

    # Normalize weights for cosine similarity computation
    W_norm = torch.nn.functional.normalize(W, p=2, dim=1)
    w_similarity = torch.mm(W_norm, W_norm.T)  # Cosine similarity matrix

    grad_norm = torch.nn.functional.normalize(grad, p=2, dim=1)
    grad_similarity = torch.mm(grad_norm, grad_norm.T)

    similarity = torch.mul(w_similarity, grad_similarity)

    chnl, n = W.shape
    # print(W.shape, grad.shape)

    # v = v + a * self.dt
    
    x_r = W.unsqueeze(-1) - W.unsqueeze(-2) # (bs, chnl, n_particles, n_particles)
    v_r = grad.unsqueeze(-1) - grad.unsqueeze(-2) # (bs, chnl, n_particles, n_particles)
    v_cm = (grad.unsqueeze(-1) + grad.unsqueeze(-2)) / 2 # (bs, chnl, n_particles, n_particles)

    x_r = torch.norm(x_r, dim=0)  # (bs, n, n), distance matrix
    v_r = torch.norm(v_r, dim=0)  # (bs, n, n), relative velocity matrix
    u_x = torch.exp(-0.05 * x_r)  # distance potential, max=1, min=0

    # for each bs, find the maximum relative velocity
    # v_r_max, _ = v_r.max(dim=0, keepdim=False)  # (bs, n)
    # print(v_r_max.shape, v_r.shape)
    # v_r_max, _ = v_r_max.max(dim=0, keepdim=False)  # (bs)
    # print(v_r_max.shape, v_r_max)
    v_r_max = torch.max(v_r)
    mask = v_r / v_r_max * u_x
    # batchnorm_mask = nn.BatchNorm1d(n).to(device)
    # mask = batchnorm_mask(mask)
    collision_mask = mask > (1 -  collision_rate) # (bs, n, n), mask of particles that collide, equivalent to bernoulli(p=v_r/v_max*u_x)
    
    delta_v = torch.zeros((chnl, n, n))

    v_r_leaving = project_to_antisymmetric(v_r, chnl)

    delta_v = v_cm + v_r_leaving - grad.unsqueeze(-1)

    # v = v + torch.sum(delta_v * collision_mask.unsqueeze(1), dim=2)
    # print(delta_v.shape, collision_mask.shape, grad.shape)
    layer.weight.grad += torch.sum(delta_v * collision_mask.unsqueeze(0), dim=1) * noise_scale
    return torch.max(torch.mul(w_similarity, grad_similarity) - torch.eye(similarity.shape[1]).to(similarity.device))
    

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

def get_cos_distance(model, target_size, pathh, epochs):
    size = target_size

    # print(model.get_weights())
    W_dir = model.linear1.weight.detach().cpu().numpy().T

    W_dir = W_dir / np.reshape(np.sqrt(np.sum(W_dir**2, axis=0)), (1, size))

    cos_distance = np.zeros((size,size))
    # print(np.shape(W_dir[:,2]))
    # print(np.shape(np.sum(W_dir**2,axis=0)))
    for i in range(size):
        for j in range(size):
            cos_distance[i][j] = np.sum(W_dir[:,i] * W_dir[:,j]) / (np.sqrt(np.sum(W_dir[:,i]**2))*np.sqrt(np.sum(W_dir[:,j]**2)))


    W_now = W_dir
    W_now_lenth = np.reshape(np.sqrt(np.sum(W_now**2,axis=0)), (1,size))

    cos_distance_matrix_temp = cos_distance

    order = []
    order1 = range(size)
    k = 0
    order_temp = []
    order1 = []
    for j in range(size):
        mark = -1 
        if j != 0:
            for i in order2:
                if cos_distance_matrix_temp[k][i] > 0.6:
                    order.append(i)
                else:
                    order1.append(i)
                if cos_distance_matrix_temp[k][i] < -0.6:
                    mark = i
        else:
            for i in range(size):
                if cos_distance_matrix_temp[0][i] > 0.6:
                    order.append(i)
                else:
                    order1.append(i)
                if cos_distance_matrix_temp[k][i] < -0.6:
                    mark = i

        order_temp = order_temp + order
        if len(order_temp) == size:
            break
        #print(len(order_temp))
        if mark == -1:
            k = order1[0]
        else:
            k = mark
        order2 = order1
        order1 = []
        order = []

    cos_distance_matrix_temp = cos_distance_matrix_temp[order_temp,:]
    cos_distance_matrix_temp = cos_distance_matrix_temp[:,order_temp]
    W_now_lenth = W_now_lenth[:,order_temp]

    record = []
    # for i in range(np.shape(W_now_lenth)[0]):
    #     if W_now_lenth[i,:]<0.00000000000000001:
    #         record.append(i)

    cos_distance_matrix_temp = np.delete(cos_distance_matrix_temp,record,axis=1)
    cos_distance_matrix_temp = np.delete(cos_distance_matrix_temp,record,axis=0)

    plt.rcParams['savefig.dpi'] = 200 #图片像素
    plt.rcParams['figure.dpi'] = 200 #分辨率
    fig,ax = plt.subplots()
    # ax = sns.heatmap(cos_distance_matrix_temp,linewidths = 0,vmin=-1,vmax=1,cmap='YlGnBu_r') # ,xticklabels = np.arange(40),yticklabels = np.arange(40))
    # ax.set_xticks(np.arange(40)) #设置x轴刻度
    # ax.set_yticks(np.arange(40)) #设置y轴刻度
    # ax.xaxis.set_ticks_position('top')
    # ax.set_xticklabels(range(40),fontsize=5)
    # ax.set_yticklabels(range(40),fontsize=5)
    plt.imshow(cos_distance_matrix_temp, cmap='YlGnBu_r')
    cb = plt.colorbar(ticks=[-1.0,-0.5,0.0,0.5,1.0])
    cb.ax.tick_params(labelsize=14)
    # ax.xaxis.set_ticks_position('top')
    plt.clim(-1, 1)
    plt.xlabel(r'Neu index',fontsize=18,labelpad=-17.0)
    plt.ylabel(r'Neu index',fontsize=18,rotation=90,labelpad=-15.0)
    # plt.colorbar(fig, ) 
    plt.xticks([0,np.shape(cos_distance_matrix_temp)[0]-1])  #去掉x轴
    plt.yticks([np.shape(cos_distance_matrix_temp)[0]-1])  #去掉y轴
    ax.set_title("$D(u,v)$",fontsize=18)
    ax.invert_yaxis()
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(r'%s/heatmap_step_%d.png'%(pathh, epochs))
    plt.close()
    plt.clf()



    # fig,ax = plt.subplots()
    # plt.plot(np.floor(np.linspace(start=1,stop=size,endpoint=True,num=size)),np.transpose(W_now_lenth))
    # plt.xticks([])  #去掉x轴
    # my_x_ticks = np.floor(np.linspace(start=1,stop=size,endpoint=True,num=size))
    # plt.xticks(my_x_ticks)
    # plt.xlabel(r'Index',fontsize=18)
    # plt.ylabel(r'Amplitude',fontsize=18)
    # plt.tick_params(axis='y',which='major',labelsize=14)
    # plt.tick_params(axis='x',which='major',labelsize=7)
    # my_x_ticks = [1,size]
    # plt.xticks(my_x_ticks)
    # ax.set_title("Amplitude: $x^{2}tanh(x)$",fontsize=18)
    # plt.tight_layout()
    # plt.savefig(r'%s/length_step_%d.png'%(pathh,epochs))
    # plt.close()
    # plt.clf()