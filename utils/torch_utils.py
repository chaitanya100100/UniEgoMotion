# import contextlib
# import inspect
# import logging
# import sys
# import time
# import typing
import math
from functools import partial
from typing import (
    Sequence,
    Union,
)

import numpy as np
import torch
import torch as th
from torch.utils.data import default_collate


def round_up_to_even(f: int) -> int:
    return math.ceil(f / 2.0) * 2


def dcn(a):
    """Convert a torch tensor to numpy array."""
    if isinstance(a, np.ndarray):
        return a
    return a.detach().cpu().numpy()


def recursive_apply(x, func):
    """Apply func to all elements of x. Works recursively on nested lists and dicts."""
    if isinstance(x, dict):
        return {k: recursive_apply(v, func) for k, v in x.items()}
    elif isinstance(x, list):
        return [recursive_apply(i, func) for i in x]
    elif isinstance(x, tuple):
        return tuple([recursive_apply(i, func) for i in x])
    elif callable(x):
        return x
    else:
        return func(x)


def _to_tensor_func(x):
    if isinstance(x, str):
        return x
    elif isinstance(x, torch.Tensor):
        return x.float()
    elif isinstance(x, (int, float, bool)):
        return x
    elif isinstance(x, np.ndarray):
        if x.dtype.type != np.str_:
            x = torch.from_numpy(x)
            if x.dtype == torch.double:
                x = x.float()
            return x
        else:
            return x
    else:
        return x
        return torch.tensor(x).float()


def _to_numpy_func(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


def _to_device_func(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x


def to_tensor(x):
    """Convert all elements of x to torch tensors. Works recursively on nested lists and dicts."""
    return recursive_apply(x, _to_tensor_func)


def to_numpy(x):
    """Convert all elements from pytorch tensors to numpy arrays. Works recursively on nested lists and dicts."""
    return recursive_apply(x, _to_numpy_func)


def to_device(x, device):
    """Move all elements of x to the specified device. Works recursively on nested lists and dicts."""
    return recursive_apply(x, partial(_to_device_func, device=device))


def print_dict(d, prefix=""):
    for k, v in d.items():
        kk = prefix + (k if isinstance(k, str) else str(k))
        if isinstance(v, dict):
            print_dict(v, kk + "/")
        elif isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
            print(kk, v)
        elif isinstance(v, list):
            print(kk, len(v))
        elif isinstance(v, (torch.Tensor, np.ndarray)):
            print(kk, v.dtype, v.shape)
        else:
            print(kk, type(v))


def careful_collate_fn(batch):

    # Avoid stacking certain keys
    avoid_keys = []
    nonstack = {}
    avail_keys = list(batch[0].keys())
    for key in avail_keys:
        if key in avoid_keys:
            nonstack[key] = [to_tensor(item.pop(key)) for item in batch]

    batch = default_collate(batch)
    batch.update(nonstack)
    return batch
