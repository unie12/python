import os
import torch
import torchrec
import torch.distributed as dist

# 분산 설정

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

dist.init_process_group(backend="nccl")