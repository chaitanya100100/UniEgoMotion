import os
import torch
import numpy as np
import copy
from tqdm.auto import tqdm
import joblib
from loguru import logger
import argparse

from dataset.smpl_utils import get_smpl, evaluate_smpl
from eval.metrics import compute_metrics
from utils.torch_utils import to_device, to_tensor, to_numpy


def get_clip_metrics(gt_aria_traj_T, pred_aria_traj_T, gt_smpl_params, pred_smpl_params, smpl, st=None, en=None):
    # deepcopy everything
    gt_aria_traj_T = copy.deepcopy(gt_aria_traj_T)
    pred_aria_traj_T = copy.deepcopy(pred_aria_traj_T)
    gt_smpl_params = copy.deepcopy(gt_smpl_params)
    pred_smpl_params = copy.deepcopy(pred_smpl_params)

    if gt_aria_traj_T is not None:
        gt_aria_traj_T = gt_aria_traj_T[st:en]
    if pred_aria_traj_T is not None:
        pred_aria_traj_T = pred_aria_traj_T[st:en]

    gt_smpl_params = {k: v[st:en] for k, v in gt_smpl_params.items()}
    pred_smpl_params = {k: v[st:en] for k, v in pred_smpl_params.items()}

    gt_kp3d, gt_verts, gt_full_pose = evaluate_smpl(smpl, gt_smpl_params)
    mdata_i = {
        "kp3d": gt_kp3d,
        "verts": gt_verts,
        "full_pose": gt_full_pose,
        "smpl_params": gt_smpl_params,
        "aria_traj_T": gt_aria_traj_T,
    }

    pred_kp3d, pred_verts, pred_full_pose = evaluate_smpl(smpl, pred_smpl_params)
    pred_mdata_i = {
        "kp3d": pred_kp3d,
        "verts": pred_verts,
        "full_pose": pred_full_pose,
        "smpl_params": pred_smpl_params,
        "aria_traj_T": pred_aria_traj_T,
    }

    mdata_i = to_device(mdata_i, torch.device("cpu"))
    pred_mdata_i = to_device(pred_mdata_i, torch.device("cpu"))

    metrics = compute_metrics(mdata_i, pred_mdata_i, smpl)
    return metrics


def main(exp_path, task, eval_suffix, data_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Experiment path: {exp_path}")

    # save path
    save_path = f"{exp_path}/metrics_3d_ee4d_{task}{eval_suffix}.pkl"
    if os.path.exists(save_path):
        print(f"Metrics already computed at {save_path}")
        return

    smpl = get_smpl()
    gt = joblib.load(f"{data_dir}/uniegomotion/ee_val_gt_for_evaluation.pkl")
    gt = to_device(gt, device)
    smpl = smpl.to(device)

    preds = joblib.load(f"{exp_path}/preds_ee4d_{task}{eval_suffix}.pkl")
    preds = to_device(preds, device)
    assert not os.path.exists(save_path), f"File already exists: {save_path}"

    all_metrics = None
    for seq_k in tqdm(list(gt.keys())):

        gt_aria_traj_T = gt[seq_k]["aria_traj_T"]
        gt_smpl_params = gt[seq_k]["smpl_params"]
        nf = len(gt_smpl_params["global_orient"])

        pred_aria_traj_T = preds[seq_k]["aria_traj_T"]
        pred_smpl_params = preds[seq_k]["smpl_params"]
        if pred_aria_traj_T is None:
            pred_aria_traj_T = gt_aria_traj_T.clone()

        metrics = get_clip_metrics(
            gt_aria_traj_T, pred_aria_traj_T, gt_smpl_params, pred_smpl_params, smpl, st=0, en=nf
        )

        # metrics of every 20 frames
        for seg_start in range(0, nf, 20):
            seg_end = min(seg_start + 20, nf)
            metrics_seg = get_clip_metrics(
                gt_aria_traj_T, pred_aria_traj_T, gt_smpl_params, pred_smpl_params, smpl, st=seg_start, en=seg_end
            )
            for k, v in metrics_seg.items():
                metrics[k + f"_seg_{seg_start}"] = v

        if all_metrics is None:
            all_metrics = {k: [] for k in metrics.keys()}
        for k, v in metrics.items():
            all_metrics[k].append(v)
    all_metrics = {k: np.array(v) for k, v in all_metrics.items()}

    assert not os.path.exists(save_path), "Metrics file already exists"
    if not os.path.exists(save_path):
        joblib.dump(all_metrics, save_path)
    logger.info(f"Metrics saved at {save_path}")


if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", type=str, default="/vision/u/chpatel/data/egoexo4d_ee4d_motion")
    parser.add_argument("--EXP_PATH", type=str, required=True)
    parser.add_argument("--EVAL_SUFFIX", type=str, default="")
    parser.add_argument("--EVAL_TASK", type=str, required=True)
    args = parser.parse_args()

    assert os.path.exists(args.EXP_PATH), f"Experiment path does not exist: {args.EXP_PATH}"
    assert args.EVAL_TASK in ["recon", "gen", "fore"], f"Task {args.EVAL_TASK} not supported"
    main(args.EXP_PATH, args.EVAL_TASK, args.EVAL_SUFFIX, args.DATA_DIR)
