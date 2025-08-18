
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F
import deepspeed
print("importing DDP and transformers modules successfully.")
print(f"Running on server node: {torch.distributed.get_rank() if torch.distributed.is_initialized() else 'Not initialized'}")