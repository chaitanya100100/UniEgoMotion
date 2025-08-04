import copy
import os
import sys

import IPython
import joblib
import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from tqdm.auto import tqdm

from config.defaults import get_cfg
from dataset.ee4d_motion_dataset import EE4D_Motion_DataModule, careful_collate_fn
from dataset.ee4d_motion_dataset import EE4D_Motion_Dataset
from module.uem_module import UEM_Module, UEM_Module_TwoStage
from utils.torch_utils import to_device

# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
# torch.multiprocessing.set_sharing_strategy("file_system")
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    device = torch.device("cuda")

    pl.seed_everything(62, workers=True)
    sys.argv = sys.argv + [
        "TRAIN.ONLY_VALIDATE",
        "True",
    ]
    cfg = get_cfg()
    assert cfg.TRAIN.EXP_PATH is not None
    assert os.path.exists(cfg.TRAIN.EXP_PATH)
    assert cfg.TRAIN.EVAL_TASK in ["recon", "gen", "fore"]
    ds_name = "ee4d"
    split = "val"

    # save path
    save_path = f"{cfg.TRAIN.EXP_PATH}/preds_{ds_name}_{cfg.TRAIN.EVAL_TASK}{cfg.TRAIN.EVAL_SUFFIX}.pkl"
    if os.path.exists(save_path):
        print(f"Preds already computed at {save_path}")
        return

    # dataset
    ds_class = {"ee4d": EE4D_Motion_Dataset}[ds_name]
    ds = ds_class(
        data_dir=cfg.DATA.DATA_DIR,
        split=split,
        repre_type=cfg.DATA.REPRE_TYPE,
        cond_img_feat=cfg.DATA.COND_IMG_FEAT,
        cond_traj=cfg.DATA.COND_TRAJ,
        window=cfg.DATA.WINDOW,
        img_feat_type=cfg.DATA.IMG_FEAT_TYPE,
        cond_betas=cfg.DATA.COND_BETAS,
    )

    is_twostage = cfg.MODEL.TRAJ_CKPT_PATH is not None

    # Model with checkpoint
    if not is_twostage:
        ckpt_path = cfg.MODEL.CKPT_PATH
        if cfg.MODEL.CKPT_PATH == "last_ckpt":
            ckpt_path = os.path.join(cfg.TRAIN.EXP_PATH, "last.ckpt")
        assert os.path.exists(ckpt_path), f"Checkpoint path {ckpt_path} does not exist"
        logger.info(f"Loading model from {ckpt_path}")
        model = UEM_Module.load_from_checkpoint(ckpt_path, cfg=cfg, map_location="cpu").to(device).eval()
    else:
        model = UEM_Module_TwoStage(cfg=cfg).to(device).eval()

    # --------------------------------------------
    # Batch processing
    # --------------------------------------------
    def process_batch(pred_batch, all_preds):
        B = len(pred_batch)
        pred_batch = careful_collate_fn(pred_batch)

        with torch.inference_mode():
            y = to_device(pred_batch["y"], device)
            x = model.sample(y, B, cond_scale=cfg.TRAIN.COND_SCALE, return_all_pred_xstart=False)
            if not is_twostage:
                if cfg.MODEL.LEARN_TRAJ:
                    pred_batch["pred"]["traj"] = to_device(x, "cpu")
                else:
                    pred_batch["pred"]["motion"] = to_device(x, "cpu")
            else:
                pred_batch["pred"]["traj"] = to_device(x[0], "cpu")
                pred_batch["pred"]["motion"] = to_device(x[1], "cpu")

        pred_mdata = ds.ret_to_full_sequence(pred_batch)
        pred_mdata = to_device(pred_mdata, "cpu")

        for i in range(B):
            seq_name = pred_batch["misc"]["seq_name"][i]
            start_idx = pred_batch["misc"]["start_idx"][i] // 3
            k = f"{seq_name}_start_{start_idx}"
            assert k not in all_preds
            all_preds[k] = {
                "smpl_params": pred_mdata["smpl_params_full"][i],
                "aria_traj_T": pred_mdata["aria_traj_T"][i],
            }
        return all_preds

    # --------------------------------------------
    # --------------------------------------------

    all_preds = {}
    batch_size = 64
    batch = []

    # We evaluate every 10th sample to make evaluation manageable.
    # Considering the segment_stride of 2 seconds, this will evaluate one 8 second sample every 20 seconds (200 frames).
    for idx in tqdm(range(0, len(ds), 10)):
        sample = ds[idx]
        sample = ds.process_sample_for_task(sample, cfg.TRAIN.EVAL_TASK)
        batch.append(sample)

        if len(batch) >= batch_size:
            all_preds = process_batch(batch, all_preds)
            batch = []

    if len(batch) > 0:
        all_preds = process_batch(batch, all_preds)
        batch = []

    # dump metric data
    joblib.dump(all_preds, save_path)
    logger.info(f"Preds saved at {save_path}")
    # IPython.embed()


if __name__ == "__main__":
    main()
