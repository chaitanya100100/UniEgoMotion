from loguru import logger
from yacs.config import CfgNode
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from mydiffusion import gaussian_diffusion as gd
from mydiffusion.respace import SpacedDiffusion, space_timesteps


class CosineAnnealingLRWithWarmup(CosineAnnealingLR):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, warmup_steps=0):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                (base_lr - self.eta_min) * self.last_epoch / self.warmup_steps + self.eta_min
                for base_lr in self.base_lrs
            ]
        else:
            return super().get_lr()


def cfg_to_dict(cfg):
    cfg_dict = {}
    for k, v in cfg.items():
        if isinstance(v, CfgNode):
            cfg_dict[k] = cfg_to_dict(v)
        else:
            cfg_dict[k] = v
    return cfg_dict


def create_gaussian_diffusion(cfg):
    predict_xstart = cfg.MODEL.PREDICT_XSTART
    steps = cfg.MODEL.DIFFUSION_STEPS
    timestep_respacing = ""  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    if not predict_xstart:
        logger.warning("Diffusion model predicts noise, not x_start.")

    betas = gd.get_named_beta_schedule(cfg.MODEL.NOISE_SCHEDULE, steps)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X),
        model_var_type=(gd.ModelVarType.FIXED_SMALL if not learn_sigma else gd.ModelVarType.LEARNED_RANGE),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        # CAN PASS OTHER PARAMETERS HERE
    )
