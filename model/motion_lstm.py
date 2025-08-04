import copy
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


class Motion_LSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.img_feat_type = cfg.DATA.IMG_FEAT_TYPE
        self.cond_img_feat = cfg.DATA.COND_IMG_FEAT
        self.lstm_type = cfg.MODEL.LSTM_TYPE
        logger.warning(f"LSTM type: {self.lstm_type}")

        assert self.lstm_type in ["gen", "fore"]

        self.repre_type = cfg.DATA.REPRE_TYPE
        self.input_feats = {"v1_beta": 234, "v4_beta": 243, "v5_beta": 243}[self.repre_type]
        traj_dim = {"v1_beta": 9, "v4_beta": 18, "v5_beta": 18}[self.repre_type]
        self.latent_dim = 1024

        # if cfg.MODEL.LEARN_TRAJ:
        #     self.input_feats = traj_dim
        #     logger.warning("LEARNING TRAJ...")

        self.img_feat_dim = 768
        if self.img_feat_type in ["dinov2", "dinov2_reg"]:
            self.img_feat_dim = 1024

        self.input_process = nn.Linear(self.input_feats, self.latent_dim)
        self.embed_traj_cond = nn.Linear(traj_dim, self.latent_dim)
        self.embed_clip_cond = nn.Linear(self.img_feat_dim, self.latent_dim)

        self.mask_tokens = nn.ParameterDict(
            {
                "traj": nn.Parameter(torch.randn(self.latent_dim) * 0.05),
                "clip": nn.Parameter(torch.randn(self.latent_dim) * 0.05),
                "motion": nn.Parameter(torch.randn(self.latent_dim) * 0.05),
            }
        )

        # bidirectional LSTM
        self.lstm = nn.LSTM(self.latent_dim, self.latent_dim, num_layers=3, bidirectional=False, batch_first=True)

        self.output_process = nn.Linear(self.latent_dim, self.input_feats)

    def mask_cond(self, cond, cond_type, cond_mask=None):
        # cond: B x T x ... x D

        if cond_mask is not None:
            cond = mask_it(cond_mask, cond, self.mask_tokens[cond_type])
            return cond

        inp_shape = cond.shape
        assert cond_type in ["traj", "clip", "motion"]

        # mask everything
        mask = torch.ones(*cond.shape[:2], device=cond.device)  # B x T

        # generation based on first frame image
        if self.lstm_type == "gen":
            if cond_type == "traj":
                pass
            elif cond_type == "clip":
                mask[:, 0] = 0  # do not mask the first frame
            elif cond_type == "motion":
                pass

        # prediction based on past images and trajectories
        elif self.lstm_type in ["fore"]:

            avail = self.cfg.DATA.WINDOW // 4
            if cond_type != "motion":
                mask[:, :avail] = 0  # do not mask the first avail frames
            else:
                pass

        cond = mask_it(mask, cond, self.mask_tokens[cond_type])

        assert cond.shape == inp_shape
        return cond

    def forward(self, x, y):
        """
        x_t: B x T x F
        y: dict
        """

        if "repaint_mask" in y:
            repaint_mask = y["repaint_mask"]
            repaint_value = y["repaint_value"]
            x = x * (1 - repaint_mask) + repaint_mask * repaint_value

        B, T, F = x.shape
        x = self.input_process(x)  # B x T x D
        x = self.mask_cond(x, "motion")  # B x T x D

        traj_mask = y["traj_mask"] if "traj_mask" in y else None
        img_mask = y["img_mask"] if "img_mask" in y else None

        if "traj" in y:  # B x T x F
            enc_traj = self.embed_traj_cond(y["traj"])  # B x T x D
            enc_traj = self.mask_cond(enc_traj, "traj", traj_mask)
            x = x + enc_traj
        else:
            x = x + self.mask_tokens["traj"]

        if "img_embs" in y:
            enc_imgs = self.embed_clip_cond(y["img_embs"])
            enc_imgs = self.mask_cond(enc_imgs, "clip", img_mask)
        else:  # if self.cond_img_feat:  # Model was trained with imgs feats. Use mask tokens here.
            enc_imgs = self.mask_tokens["clip"].view(1, 1, -1).repeat(B, T, 1)
        x = x + enc_imgs

        lens = y["valid_frames"].sum(-1).cpu()  # B
        x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=T)

        x = self.output_process(x)  # B x T x (J x F)
        x = x.view(B, T, F)

        if "repaint_mask" in y:
            assert self.cfg.MODEL.PREDICT_XSTART
            repaint_mask = y["repaint_mask"]
            repaint_value = y["repaint_value"]
            x = x * (1 - repaint_mask) + repaint_mask * repaint_value

        return x


if __name__ == "__main__":
    cfg = lambda: None
    cfg.DATA = lambda: None
    cfg.MODEL = lambda: None
    cfg.DATA.COND_IMG_FEAT = True
    cfg.DATA.IMG_FEAT_TYPE = "dinov2"
    cfg.DATA.REPRE_TYPE = "v1"
    cfg.MODEL.LSTM_TYPE = "gen"
    cfg.DATA.WINDOW = 10

    model = Motion_LSTM(cfg)
    x = torch.randn(2, 10, 224)

    y = {
        "traj": torch.randn(2, 10, 9),
        "img_embs": torch.randn(2, 10, 1024),
        "valid_frames": torch.ones(2, 10).long(),
        "valid_img_embs": torch.ones(2, 10).long(),
    }
    out = model(x, y)

    print(sum(p.numel() for p in model.parameters()) / 1e6)

    import IPython

    IPython.embed()
