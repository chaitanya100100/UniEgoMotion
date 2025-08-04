import os
import pytorch_lightning as pl
import torch
import torch.distributed
from loguru import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from model.uniegomotion import UniEgoMotion
from model.motion_lstm import Motion_LSTM
from model.motion_unet import Motion_Unet

from module.utils import cfg_to_dict, create_gaussian_diffusion
from mydiffusion.resample import create_named_schedule_sampler
from mydiffusion.gaussian_diffusion import sum_flat


class UEM_Module(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        if not cfg.TRAIN.ONLY_VALIDATE:
            self.save_hyperparameters(cfg_to_dict(cfg))
        self.cfg = cfg
        self.window = self.cfg.DATA.WINDOW
        self.model_name = cfg.MODEL.MODEL_NAME
        self.learn_traj = cfg.MODEL.LEARN_TRAJ
        if self.learn_traj:
            assert not self.cfg.DATA.COND_TRAJ

        self.is_lstm = False
        if self.model_name == "uem":
            self.model = UniEgoMotion(self.cfg)
        elif self.model_name == "lstm":
            self.model = Motion_LSTM(self.cfg)
            self.is_lstm = True
        elif self.model_name == "unet":
            self.model = Motion_Unet(self.cfg)
        else:
            raise ValueError(f"Unknown model name {self.model_name}")

        if self.is_lstm:
            return

        self.diffusion = create_gaussian_diffusion(cfg)

        self.schedule_sampler_type = "uniform"
        # self.schedule_sampler_type = "loss-second-moment"
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion)
        self.last_iters = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
            # fused=True,
        )
        # return optimizer

        # scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=15, verbose=True)
        scheduler = StepLR(optimizer, self.cfg.TRAIN.NUM_EPOCHS - 50, gamma=0.1)
        return [optimizer], [
            {
                "scheduler": scheduler,
                # "monitor": "val/loss_epoch",
                "interval": "epoch",
            }
        ]

    def on_train_start(self):
        if self.cfg.MODEL.CKPT_PATH is None:
            return
        if self.cfg.TRAIN.USE_CKPT_LR:
            logger.warning("Using LR from checkpoint.")
            return
        logger.warning("Discarding LR of optimizer dict and using config LR.")
        for g in self.optimizers().param_groups:
            g["lr"] = self.cfg.TRAIN.LR
        for g in self.optimizers().param_groups:
            g["weight_decay"] = self.cfg.TRAIN.WEIGHT_DECAY

    def training_step(self, batch, batch_idx, mode="train"):
        if self.is_lstm:
            x = batch["misc"]["traj"] if self.learn_traj else batch["misc"]["motion"]
            y = batch["y"]
            pred_x = self.model(x.clone(), y)
            target = x

            mask = y["valid_frames"]
            mask = mask.view(list(mask.shape) + [1] * (x.ndim - mask.ndim))
            mask = mask.expand_as(target)
            loss_here = sum_flat(((target - pred_x) * mask) ** 2)
            denom = sum_flat(mask)
            loss = (loss_here / denom).mean()
            self.log(f"{mode}/loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=x.shape[0])
            return loss

        x = batch["misc"]["traj"] if self.learn_traj else batch["misc"]["motion"]
        t, weights = self.schedule_sampler.sample(x.shape[0], self.device)
        losses = self.diffusion.training_losses(self.model, x, t, model_kwargs={"y": batch["y"]})
        if mode == "train" and self.schedule_sampler_type != "uniform":
            self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

        loss = (losses["loss"] * weights).mean()

        # To monitor diffusion step wise loss
        assert losses["loss"].shape[0] == x.shape[0]
        diff_steps, bin_size = self.cfg.MODEL.DIFFUSION_STEPS, self.cfg.MODEL.DIFFUSION_STEPS // 10
        for idx, i in enumerate(range(0, diff_steps, bin_size)):
            mask = (t >= i) & (t < i + bin_size)
            if mask.any():
                range_loss = losses["loss"][mask].mean()
                self.log(
                    f"{mode}_loss/{idx}",
                    range_loss,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=mask.sum(),
                )

        for k, v in losses.items():
            self.log(f"{mode}/{k}", v.mean(), on_step=True, on_epoch=True, sync_dist=True, batch_size=x.shape[0])

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.training_step(batch, batch_idx, mode="val")

    def sample(self, y, B=1, cond_scale=None, return_all_pred_xstart=False):
        if self.is_lstm:
            x = torch.zeros(B, self.window, self.model.input_feats, device=self.device)
            x = self.model(x, y)
            return x

        # B = y["traj"].shape[0]
        for k, v in y.items():
            assert len(v) == B, f"y[{k}] has batch size {len(v)} but expected {B}"
        x = self.diffusion.p_sample_loop(
            self.model,
            (B, self.window, self.model.input_feats),
            model_kwargs={"y": y, "cond_scale": cond_scale, "diffusion": self.diffusion},
            clip_denoised=False,
            noise=None,
            progress=False,
            # skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            # init_image=None,
            # dump_steps=None,
            # const_noise=False,
            return_all_pred_xstart=return_all_pred_xstart,
        )
        if return_all_pred_xstart:
            return x[0], x[1]  # final sample, all predicted xstart
        return x


class UEM_Module_TwoStage(pl.LightningModule):
    """For two stage baseline."""

    def __init__(self, cfg):
        super().__init__()

        traj_ckpt_path = cfg.MODEL.TRAJ_CKPT_PATH
        traj_exp_path = os.path.dirname(traj_ckpt_path)
        traj_cfg = cfg.clone()
        traj_cfg.defrost()
        traj_cfg.merge_from_file(f"{traj_exp_path}/hparams.yaml")
        traj_cfg.freeze()

        motion_ckpt_path = cfg.MODEL.MOTION_CKPT_PATH
        motion_exp_path = os.path.dirname(motion_ckpt_path)
        motion_cfg = cfg.clone()
        motion_cfg.defrost()
        motion_cfg.merge_from_file(f"{motion_exp_path}/hparams.yaml")
        motion_cfg.freeze()

        logger.warning(f"Loading from {traj_ckpt_path}")
        self.traj_module = UEM_Module.load_from_checkpoint(traj_ckpt_path, cfg=traj_cfg, map_location="cpu")

        logger.warning(f"Loading from {motion_ckpt_path}")
        self.motion_module = UEM_Module.load_from_checkpoint(motion_ckpt_path, cfg=motion_cfg, map_location="cpu")

        self.motion_cfg = motion_cfg
        self.traj_cfg = traj_cfg

    def sample_traj(self, y, B=1, cond_scale=None, return_all_pred_xstart=False):
        return self.traj_module.sample(y, B, cond_scale, return_all_pred_xstart)

    def sample_motion(self, y, B=1, cond_scale=None, return_all_pred_xstart=False):
        return self.motion_module.sample(y, B, cond_scale, return_all_pred_xstart)

    def sample(self, y, B=1, cond_scale=None, return_all_pred_xstart=False):
        assert self.motion_cfg.DATA.REPRE_TYPE == self.traj_cfg.DATA.REPRE_TYPE

        # prep for trajectory prediction
        traj = y.pop("traj", None)  # B x T x D
        traj_mask = y.pop("traj_mask", None)  # B x T

        assert traj is not None
        if traj_mask is None:
            traj_mask = torch.zeros_like(traj[:, :, 0])

        y["repaint_mask"] = 1 - traj_mask[..., None].expand_as(traj)
        y["repaint_value"] = traj
        traj_pred = self.sample_traj(y, B, cond_scale, return_all_pred_xstart=False)

        # prep for motion prediction
        y.pop("repaint_mask", None)
        y.pop("repaint_value", None)
        y["traj"] = traj_pred
        y["traj_mask"] = torch.zeros_like(traj_mask)

        motion = self.sample_motion(y, B, cond_scale, return_all_pred_xstart=False)
        return traj_pred, motion
