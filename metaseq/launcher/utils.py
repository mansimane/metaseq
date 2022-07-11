import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import torch

import subprocess
from time import sleep

def get_gpu_memory_map_smi():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def get_gpu_memory_map_torch():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """


    alloced = torch.cuda.memory_allocated(device=0)
    max_alloced = torch.cuda.max_memory_allocated(device=0)

    # convert to GB for printing
    alloced /= 1024**2
    max_alloced /= 1024**3

    return alloced, max_alloced

class MemoryMonitor:
    def __init__(self):
        self.keep_measuring = True

    def measure_usage(self):
        count = 0
        max_usage = 0
        max_usage_torch_max_alloced = 0
        max_usage_torch_alloced = 0
        total = 0.0
        total_torch_max_alloced = 0.0
        total_torch_alloced = 0.0


        while self.keep_measuring:
            usage = list(get_gpu_memory_map_smi().values())[0]

            # gpu_average = sum(usage)/len(usage)
            count += 1.0
            total += usage
            max_usage = max(
                max_usage,
                usage
            )

            # torch usage
            usage_torch_alloced, usage_torch_max_alloced = get_gpu_memory_map_torch()
            total_torch_alloced += usage_torch_alloced
            total_torch_max_alloced += usage_torch_max_alloced
            max_usage_torch_alloced= max(max_usage_torch_alloced, usage_torch_alloced)
            max_usage_torch_max_alloced= max(max_usage_torch_max_alloced, usage_torch_max_alloced)
            sleep(2)

        return total/count, max_usage, total_torch_alloced/count, total_torch_max_alloced/count,  max_usage_torch_alloced, max_usage_torch_max_alloced