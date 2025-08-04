import copy
import numpy as np
import torch
import torch.nn as nn
from model.core import DecoderBlock
from model.core import EncoderBlock
from loguru import logger


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


class UniEgoMotion(nn.Module):
    def __init__(
        self,
        cfg,
        dropout=0.1,
    ):
        super().__init__()

        self.cfg = cfg

        latent_dim = 768
        ff_size = 768 * 2
        num_layers = 12
        num_heads = 12

        self.img_feat_type = cfg.DATA.IMG_FEAT_TYPE
        self.cond_img_feat = cfg.DATA.COND_IMG_FEAT
        self.cond_betas = cfg.DATA.COND_BETAS
        self.encoder_tsfm = cfg.MODEL.ENCODER_TSFM
        self.finetune_type = cfg.MODEL.FINETUNE_TYPE

        if self.finetune_type is not None:
            assert self.finetune_type in ["gen", "fore", "recon"]
            logger.warning(f"Using finetune type {self.finetune_type}.")

        self.repre_type = cfg.DATA.REPRE_TYPE
        self.input_feats = {"v1_beta": 234, "v4_beta": 243, "v5_beta": 243}[self.repre_type]
        traj_dim = {"v1_beta": 9, "v4_beta": 18, "v5_beta": 18}[self.repre_type]
        self.latent_dim = latent_dim

        if cfg.MODEL.LEARN_TRAJ:
            self.input_feats = traj_dim
            logger.warning("LEARNING TRAJ... USING SMALLER MODEL.")
            latent_dim = 512
            ff_size = 512 * 2
            num_layers = 8
            num_heads = 8

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.img_feat_dim = 768
        if self.img_feat_type in ["dinov2", "dinov2_reg"]:
            self.img_feat_dim = 1024
        if self.img_feat_type in ["egovideo"]:
            self.img_feat_dim = 512

        # self.cond_mask_prob = cond_mask_prob
        self.cond_mask_prob = {
            "traj": 0.5,
            "clip": 0.1,
            "subseq_frames": 0.5,
        }

        self.input_process = nn.Linear(self.input_feats, self.latent_dim)
        self.pos_enc = PositionalEncoding(self.latent_dim, self.dropout)
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

        self.zero_mask_token = cfg.MODEL.ZERO_MASK_TOKEN
        if self.zero_mask_token:
            logger.warning("Using zero mask token.")
            del self.mask_tokens
            self.mask_tokens = nn.ParameterDict(
                {
                    "traj": nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False),
                    "clip": nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False),
                    "text": nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False),  # Not used
                }
            )

        if self.encoder_tsfm is not None:
            assert self.encoder_tsfm in ["add"]
            logger.warning("Using encoder.")
            self.tsfm = nn.ModuleList(
                [EncoderBlock(self.latent_dim, self.num_heads, self.dropout, 2) for _ in range(self.num_layers)]
            )
        else:
            self.tsfm = nn.ModuleList(
                [DecoderBlock(self.latent_dim, self.num_heads, self.dropout, 2) for _ in range(self.num_layers)]
            )

        self.output_process = nn.Linear(self.latent_dim, self.input_feats)

    def mask_cond_finetune(self, cond, cond_type, cond_mask=None):
        # cond: B x T x ... x D

        if cond_mask is not None:
            cond = mask_it(cond_mask, cond, self.mask_tokens[cond_type])
            return cond

        inp_shape = cond.shape
        assert cond_type in ["traj", "clip", "motion"]

        # For recon, do not mask anything
        if self.finetune_type == "recon":
            return cond

        # mask everything
        mask = torch.ones(*cond.shape[:2], device=cond.device)  # B x T

        if self.finetune_type == "gen":
            if cond_type == "traj":
                pass
            elif cond_type == "clip":
                mask[:, 0] = 0  # do not mask the first frame
            elif cond_type == "motion":
                pass

        # prediction based on past images and trajectories
        elif self.finetune_type in ["fore"]:

            avail = self.cfg.DATA.WINDOW // 4
            if cond_type != "motion":
                mask[:, :avail] = 0  # do not mask the first avail frames
            else:
                pass

        cond = mask_it(mask, cond, self.mask_tokens[cond_type])

        assert cond.shape == inp_shape
        return cond

    def mask_cond(self, cond, cond_type, cond_mask=None):
        # cond: B x T x ... x D
        if self.finetune_type is not None:
            return self.mask_cond_finetune(cond, cond_type, cond_mask)

        # for inference of recon, fore and gen, masks are already provided.
        if cond_mask is not None:
            cond = mask_it(cond_mask, cond, self.mask_tokens[cond_type])
            return cond

        if not self.training:
            # technically, this should never happen because we always do inference with cond_mask.
            return cond
        inp_shape = cond.shape

        # During training, mask whole condition randomly based on probability.
        # To simulate generation, we additionally mask subsequent frames (after first frame) with some probability.
        # Overall, this will simulate recon and gen tasks with some probability.

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

    def pos_enc_and_process_img_feat(self, x, enc_imgs):
        if self.img_feat_type in ["dinov2_reg"]:
            # B x T x 5 x D
            for i in range(5):
                enc_imgs[:, :, i] = self.pos_enc(enc_imgs[:, :, i])
            return enc_imgs
        if self.img_feat_type in ["clip_all", "dinov2", "egovideo"]:
            enc_imgs = self.pos_enc(enc_imgs)
            return enc_imgs
        else:
            raise ValueError(f"img_feat_type is {self.img_feat_type}")

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
        x = self.pos_enc(x)  # B x T x D

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
            assert self.cond_betas
            enc_betas = self.embed_betas(y["betas"])  # B x D
            x = x + enc_betas[:, None, :]  # B x T x D

        all_cond = [enc_time[:, None]]  # B x 1 x D
        all_cond_mask = [torch.ones(B, 1, dtype=torch.long, device=x.device)]  # B x 1
        if "img_embs" in y:
            enc_imgs = self.embed_clip_cond(y["img_embs"])
            enc_imgs = self.mask_cond(enc_imgs, "clip", img_mask)
            enc_imgs = self.pos_enc_and_process_img_feat(x, enc_imgs)

            enc_img_mask = y["valid_img_embs"]  # B x C

            if self.img_feat_type == "dinov2_reg":
                enc_imgs = enc_imgs.flatten(1, 2)  # B x T x 5 x D to B x T5 x D
                enc_img_mask = enc_img_mask[:, :, None].repeat(1, 1, 5).flatten(1, 2)  # B x T to B x T5

            all_cond.append(enc_imgs)
            all_cond_mask.append(enc_img_mask)
        else:  # if self.cond_img_feat:  # Model was trained with img feats. Use mask tokens here.
            enc_imgs = self.mask_tokens["clip"].view(1, 1, -1).repeat(B, T, 1)
            enc_imgs = self.pos_enc_and_process_img_feat(x, enc_imgs)
            enc_img_mask = y["valid_frames"]

            all_cond.append(enc_imgs)
            all_cond_mask.append(enc_img_mask)

        x = torch.cat((enc_time[:, None], x), axis=1)  # B x (T+1) x D
        mask = y["valid_frames"]  # B x T where valid are 1
        mask = torch.cat((torch.ones((B, 1), device=mask.device, dtype=mask.dtype), mask), dim=1)  # B x (T+1)
        mask = mask[:, None, None, :]  # B x 1 x 1 x (T+1)

        context_mask = torch.cat(all_cond_mask, dim=1)  # B x C
        context_mask = context_mask[:, None, None, :]  # B x 1 x 1 x C
        context = torch.cat(all_cond, dim=1)  # B x C x D

        if self.encoder_tsfm:
            if self.encoder_tsfm == "add":
                x = x + context
            else:
                raise ValueError(f"encoder_tsfm is {self.encoder_tsfm}")

            for enc in self.tsfm:
                x = enc(x=x, mask=mask)
        else:
            for dec in self.tsfm:
                x = dec(x=x, context=context, mask=mask, context_mask=context_mask)

        x = self.output_process(x[:, 1 : 1 + T])  # B x T x (J x F)
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
    cfg.DATA.COND_BETAS = True
    cfg.DATA.IMG_FEAT_TYPE = "clip_all"
    cfg.DATA.REPRE_TYPE = "v1"
    cfg.MODEL.LEARN_TRAJ = False

    model = UniEgoMotion(cfg)
    x = torch.randn(2, 10, 224)
    timesteps = torch.tensor([342, 21])

    y = {
        "traj": torch.randn(2, 10, 9),
        "img_embs": torch.randn(2, 10, 768),
        "valid_frames": torch.ones(2, 10).long(),
        "valid_img_embs": torch.ones(2, 10).long(),
        "betas": torch.randn(2, 10),
    }
    out = model(x, timesteps, y)

    import IPython

    IPython.embed()
