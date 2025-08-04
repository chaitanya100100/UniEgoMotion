import copy
import numpy as np
import torch
import torch.nn as nn
from model.unet_1d import TemporalUnet
from loguru import logger
import einops


def mask_it(mask, cond, replace_token):
    # mask: B bool
    # cond: B x ... x D
    # replace_token: D
    expand_dims = [1] * (len(cond.shape) - len(mask.shape))
    mask = mask.view(*mask.shape, *expand_dims).float()  # B x ... x 1

    expand_dims = [1] * (len(cond.shape) - 1)
    mask_token = replace_token.view(*expand_dims, -1)  # 1 x ... x D
    cond = cond * (1.0 - mask) + mask * mask_token
    return cond


class Motion_Unet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.img_feat_type = cfg.DATA.IMG_FEAT_TYPE
        self.cond_img_feat = cfg.DATA.COND_IMG_FEAT
        self.cond_betas = cfg.DATA.COND_BETAS

        self.repre_type = cfg.DATA.REPRE_TYPE
        motion_dim = {"v1_beta": 234, "v4_beta": 243, "v5_beta": 243}[self.repre_type]
        traj_dim = {"v1_beta": 9, "v4_beta": 18, "v5_beta": 18}[self.repre_type]
        self.latent_dim = 512
        if cfg.MODEL.LEARN_TRAJ:
            self.input_feats = traj_dim
            self.dim_mults = [0.125, 0.25, 0.5]
            logger.warning("LEARNING TRAJ...")
        else:
            self.input_feats = motion_dim
            self.dim_mults = [2, 2, 2, 2]
            self.dim_mults = [1, 1, 1, 1]

        self.img_feat_dim = 768
        if self.img_feat_type in ["dinov2", "dinov2_reg"]:
            self.img_feat_dim = 1024

        # self.cond_mask_prob = cond_mask_prob
        self.cond_mask_prob = {
            "traj": 0.5,
            "clip": 0.1,
            "subseq_frames": 0.5,
        }

        self.input_process = nn.Linear(self.input_feats, self.latent_dim)
        self.pos_enc = PositionalEncoding(self.latent_dim, 0.1)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.pos_enc)
        self.embed_traj_cond = nn.Linear(traj_dim, self.latent_dim)
        self.embed_clip_cond = nn.Linear(self.img_feat_dim, self.latent_dim)
        self.embed_text_cond = nn.Linear(768, self.latent_dim)  # Not used

        if self.cond_betas:
            self.embed_betas = nn.Linear(10, self.latent_dim)
            logger.warning("Using betas.")

        self.mask_tokens = nn.ParameterDict(
            {
                "traj": nn.Parameter(torch.randn(self.latent_dim) * 0.05),
                "clip": nn.Parameter(torch.randn(self.latent_dim) * 0.05),
                "text": nn.Parameter(torch.randn(self.latent_dim) * 0.05),  # Not used
            }
        )

        self.unet = TemporalUnet(
            self.latent_dim, self.latent_dim, self.latent_dim, self.dim_mults, adagn=True, zero=True
        )

        self.output_process = nn.Linear(self.latent_dim, self.input_feats)

    def mask_cond(self, cond, cond_type, cond_mask=None):
        # cond: B x T x ... x D
        if cond_mask is not None:
            cond = mask_it(cond_mask, cond, self.mask_tokens[cond_type])
            return cond

        if not self.training:
            return cond
        inp_shape = cond.shape

        # Mask condition of all frames using probability
        mask = torch.rand(cond.shape[0], device=cond.device) < self.cond_mask_prob[cond_type]  # B
        cond = mask_it(mask, cond, self.mask_tokens[cond_type])

        assert cond_type in ["traj", "clip"]

        # Mask condition of subsequent frames using probability. For trajectory, we also mask the first frame.
        mask = torch.rand(cond.shape[0], device=cond.device) < self.cond_mask_prob["subseq_frames"]  # B
        if cond_type == "traj":
            cond = mask_it(mask, cond, self.mask_tokens[cond_type])
        else:
            subsequent_cond = mask_it(mask, cond[:, 1:], self.mask_tokens[cond_type])
            cond = torch.cat((cond[:, :1], subsequent_cond), axis=1)

        assert cond.shape == inp_shape
        return cond

    def forward(self, x, timesteps, y, cond_scale=None, diffusion=None):
        """
        x_t: B x T x F
        timesteps: B
        y: dict
        """
        if cond_scale is not None:
            assert diffusion is not None
            x_cond = self.forward(x, timesteps, y)  # conditional output
            x_uncond = self.forward(x, timesteps, {"valid_frames": y["valid_frames"]})  # unconditional output

            assert self.cfg.MODEL.PREDICT_XSTART
            x_scaled = x_uncond + (x_cond - x_uncond) * cond_scale
            return x_scaled

        B, T, F = x.shape
        x = self.input_process(x)  # B x T x D
        # x = self.pos_enc(x)  # B x T x D

        traj_mask = y["traj_mask"] if "traj_mask" in y else None
        img_mask = y["img_mask"] if "img_mask" in y else None

        enc_time = self.embed_timestep(timesteps)  # B x D

        if "traj" in y:  # B x T x F
            enc_traj = self.embed_traj_cond(y["traj"])  # B x T x D
            enc_traj = self.mask_cond(enc_traj, "traj", traj_mask)
            x = x + enc_traj
        else:
            x = x + self.mask_tokens["traj"]

        if "betas" in y:
            assert False
            assert self.cond_betas
            enc_betas = self.embed_betas(y["betas"])  # B x D
            x = x + enc_betas[:, None, :]  # B x T x D

        if "img_embs" in y:
            enc_imgs = self.embed_clip_cond(y["img_embs"])
            enc_imgs = self.mask_cond(enc_imgs, "clip", img_mask)
            if self.img_feat_type == "dinov2_reg":
                enc_imgs = enc_imgs.flatten(1, 2)  # B x T x 5 x D to B x T5 x D
        else:
            enc_imgs = self.mask_tokens["clip"].view(1, 1, -1).repeat(B, T, 1)
        x = x + enc_imgs

        x = einops.rearrange(x, "b t d -> t b d")
        bT = int(np.ceil(T / 16)) * 16
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, bT - T), value=0)
        x = self.unet.forward(x, enc_time)
        x = x[:T]
        x = einops.rearrange(x, "t b d -> b t d")

        x = self.output_process(x)  # B x T x (J x F)
        x = x.view(B, T, F)

        if "repaint_mask" in y:
            assert self.cfg.MODEL.PREDICT_XSTART
            repaint_mask = y["repaint_mask"]
            repaint_value = y["repaint_value"]
            x = x * (1 - repaint_mask) + repaint_mask * repaint_value

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe[None])  # 1 x Tmax x D

    def forward(self, x):
        # x is B x T x D
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, pos_enc):
        super().__init__()
        self.latent_dim = latent_dim
        self.pos_enc = pos_enc  # .pe is 1 x Tmax x D

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        # timesteps is (B,)
        t = self.pos_enc.pe[0, timesteps]  # B x D
        return self.time_embed(t)  # B x D


if __name__ == "__main__":
    cfg = lambda: None
    cfg.DATA = lambda: None
    cfg.MODEL = lambda: None
    cfg.DATA.COND_IMG_FEAT = True
    cfg.DATA.COND_BETAS = False
    cfg.DATA.IMG_FEAT_TYPE = "dinov2"
    cfg.DATA.REPRE_TYPE = "v1"
    cfg.MODEL.LEARN_TRAJ = False
    cfg.MODEL.PREDICT_XSTART = True

    model = Motion_Unet(cfg)
    x = torch.randn(2, 10, 224)
    timesteps = torch.tensor([342, 21])

    y = {
        "traj": torch.randn(2, 10, 9),
        "img_embs": torch.randn(2, 10, 1024),
        "valid_frames": torch.ones(2, 10).long(),
        "valid_img_embs": torch.ones(2, 10).long(),
        # "betas": torch.randn(2, 10),
    }
    out = model(x, timesteps, y)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
    import IPython

    IPython.embed()
