# TMR model is used to get motion features to compute semantic motion quality metrics.
# Loads this model: https://github.com/nv-tlabs/stmc/blob/main/mtt/load_tmr_model.py
# Please setup stmc repo separately and set the path in the code below.

import os
import sys
import torch
import copy
import numpy as np


# x.T will be deprecated in pytorch
def transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))


def get_sim_matrix(x, y):
    x_logits = torch.nn.functional.normalize(x, dim=-1)
    y_logits = torch.nn.functional.normalize(y, dim=-1)
    sim_matrix = x_logits @ transpose(y_logits)
    return sim_matrix


def get_tmr_model(device="cpu"):
    # print("Last 2 sys path", sys.path[-2:])
    sys.path.append("/vision/u/chpatel/stmc/TMR")

    current_dir = copy.deepcopy(os.getcwd())
    # print("Current dir", os.getcwd())
    os.chdir("/vision/u/chpatel/stmc/TMR")
    # print("Current dir", os.getcwd())

    from mtt.load_tmr_model import load_tmr_model_easy  # type: ignore

    # print("Last 2 sys path", sys.path[-2:])
    tmr_forward = load_tmr_model_easy(device)

    os.chdir(current_dir)
    # print("Current dir", os.getcwd())
    sys.path = sys.path[:-1]
    # print("Last 2 sys path", sys.path[-2:])

    return tmr_forward
