import os
import random

import numpy as np
import torch


def seed_everything(seed, deterministic=True):
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True, warn_only=True)


def make_torch_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def make_worker_init_fn(seed):
    base_seed = int(seed)

    def _init_fn(worker_id):
        worker_seed = (base_seed + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(worker_seed)

    return _init_fn
