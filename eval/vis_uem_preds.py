import os
import torch
import numpy as np
import copy
from tqdm.auto import tqdm
import joblib
from loguru import logger
import argparse


from dataset.smpl_utils import get_smpl, evaluate_smpl
from utils.torch_utils import to_device, to_tensor, to_numpy
from utils.vis_utils import visualize_sequence_blender, save_video, pad_filler, pad_filler_traj
from eval.selected_seqs import selected_seqs


def main(exp_path, task, eval_suffix="", overwrite=False, data_dir=""):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Experiment path: {exp_path}")

    # save path
    save_path = f"{exp_path}/vis_{task}{eval_suffix}"
    if os.path.exists(save_path) and not overwrite:
        print(f"Vis already done at {save_path}")
        return

    smpl = get_smpl()
    gt = joblib.load(f"{data_dir}/uniegomotion/ee_val_gt_for_evaluation.pkl")
    gt = to_device(gt, device)
    smpl = smpl.to(device)

    preds = joblib.load(f"{exp_path}/preds_ee4d_{task}{eval_suffix}.pkl")
    preds = to_device(preds, device)

    os.makedirs(save_path, exist_ok=True)

    for seq_k in tqdm(list(selected_seqs)):
        print(seq_k)

        gt_aria_traj_T = gt[seq_k]["aria_traj_T"]
        gt_smpl_params = gt[seq_k]["smpl_params"]
        nf_gt = len(gt_smpl_params["global_orient"])

        # pred_aria_traj_T = preds[seq_k]["aria_traj_T"]
        pred_smpl_params = preds[seq_k]["smpl_params"]
        nf_pred = len(pred_smpl_params["global_orient"])

        _, pred_verts, _ = evaluate_smpl(smpl, pred_smpl_params)

        if task == "recon":
            _, gt_verts, _ = evaluate_smpl(smpl, gt_smpl_params)
            gt_aria_traj_T = gt_aria_traj_T
        elif task == "gen":
            gt_verts = None
            gt_aria_traj_T = None
        elif task in ["fore"]:
            avail = min(nf_gt, 20)
            # _, gt_verts, _ = evaluate_smpl(smpl, gt_smpl_params)
            # gt_verts = gt_verts[:avail]
            # gt_verts = pad_filler(gt_verts, nf_pred)

            gt_aria_traj_T = gt_aria_traj_T[:avail]
            gt_aria_traj_T = pad_filler_traj(gt_aria_traj_T, nf_pred)

            gt_verts = None
        else:
            raise NotImplementedError

        imgs = visualize_sequence_blender(
            aria_traj=gt_aria_traj_T,
            verts=gt_verts,
            faces=smpl.faces,
            pred_verts=pred_verts,
        )

        fname = f"{seq_k}.mp4"
        save_video(imgs[..., ::-1], fname, save_path, fps=10)


if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--EXP_PATH", type=str, required=True)
    parser.add_argument("--EVAL_SUFFIX", type=str, default="")
    parser.add_argument("--EVAL_TASK", type=str, required=True)
    parser.add_argument("--OVERWRITE", action="store_true")
    parser.add_argument("--DATA_DIR", type=str, default="/vision/u/chpatel/data/egoexo4d_ee4d_motion")
    args = parser.parse_args()

    assert os.path.exists(args.EXP_PATH), f"Experiment path does not exist: {args.EXP_PATH}"
    assert args.EVAL_TASK in ["recon", "gen", "fore"], f"Task {args.EVAL_TASK} not supported"
    main(args.EXP_PATH, args.EVAL_TASK, args.EVAL_SUFFIX, args.OVERWRITE, args.DATA_DIR)
