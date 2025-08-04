import os
import argparse

import joblib
import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

from dataset.smpl_utils import evaluate_smpl, get_smpl
from eval.metrics import calculate_activation_statistics_normalized, calculate_frechet_distance
from model.tmr_eval_model import get_tmr_model
from utils.guofeats import joints_to_guofeats
from utils.torch_utils import to_device, to_numpy, to_tensor


def transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))


def get_sim_matrix(x, y):
    x_logits = torch.nn.functional.normalize(x, dim=-1)
    y_logits = torch.nn.functional.normalize(y, dim=-1)
    sim_matrix = x_logits @ transpose(y_logits)
    return sim_matrix


def upsample_in_time(x, factor=3):
    # x : T x *
    T = x.shape[0]
    i = torch.linspace(0, T - 1, (T - 1) * factor, device=x.device)
    li = torch.floor(i).clamp(0, T - 1).long()
    ri = torch.floor(i + 1).clamp(0, T - 1).long()

    a = (ri.float() - i).view(-1, *[1] * (x.ndim - 1))

    ret = x[li] * a + x[ri] * (1 - a)
    return ret


def main(exp_path, task, eval_suffix, data_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Experiment path: {exp_path}")

    # save path
    save_path = f"{exp_path}/metrics_sem_ee4d_{task}{eval_suffix}.pkl"
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

    tmr_forward = get_tmr_model(device)

    gt_embs_all = []
    pred_embs_all = []
    for seq_k in tqdm(list(gt.keys())):

        # pred_aria_traj_T = preds[seq_k]["aria_traj_T"]
        pred_smpl_params = preds[seq_k]["smpl_params"]
        pred_kp3d, pred_verts, pred_full_pose = evaluate_smpl(smpl, pred_smpl_params)

        # gt_aria_traj_T = gt[seq_k]["aria_traj_T"]
        gt_smpl_params = gt[seq_k]["smpl_params"]
        gt_kp3d, gt_verts, gt_full_pose = evaluate_smpl(smpl, gt_smpl_params)

        gt_joints = gt_kp3d[:, :22]
        pred_joints = pred_kp3d[:, :22]
        gt_joints = upsample_in_time(gt_joints.cpu())
        pred_joints = upsample_in_time(pred_joints.cpu())

        gt_guofeats = joints_to_guofeats(gt_joints)
        pred_guofeats = joints_to_guofeats(pred_joints)

        gt_emb = tmr_forward(gt_guofeats.numpy()[None])
        pred_emb = tmr_forward(pred_guofeats.numpy()[None])

        gt_embs_all.append(gt_emb.cpu())
        pred_embs_all.append(pred_emb.cpu())

    gt_embs_all = torch.cat(gt_embs_all, dim=0)  # N x 256
    pred_embs_all = torch.cat(pred_embs_all, dim=0)  # N x 256

    # semantic similarity score
    ss_score = torch.diag(get_sim_matrix(gt_embs_all, pred_embs_all)) / 2 + 0.5
    ss_score = ss_score.numpy()

    # Frechet distance
    mu_gt, sigma_gt = calculate_activation_statistics_normalized(gt_embs_all.numpy())
    mu_pred, sigma_pred = calculate_activation_statistics_normalized(pred_embs_all.numpy())
    fid_score = calculate_frechet_distance(mu_pred, sigma_pred, mu_gt, sigma_gt)

    all_metrics = {
        "semantic_similarity": ss_score,
        "fid": fid_score,
    }
    assert not os.path.exists(save_path), "Metrics file already exists"
    if not os.path.exists(save_path):
        joblib.dump(all_metrics, save_path)
    logger.info(f"Metrics saved at {save_path}")


if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--EXP_PATH", type=str, required=True)
    parser.add_argument("--EVAL_SUFFIX", type=str, default="")
    parser.add_argument("--EVAL_TASK", type=str, required=True)
    parser.add_argument("--DATA_DIR", type=str, default="/vision/u/chpatel/data/egoexo4d_ee4d_motion")
    args = parser.parse_args()

    assert os.path.exists(args.EXP_PATH), f"Experiment path does not exist: {args.EXP_PATH}"
    assert args.EVAL_TASK in ["recon", "gen", "fore"], f"Task {args.EVAL_TASK} not supported"
    main(args.EXP_PATH, args.EVAL_TASK, args.EVAL_SUFFIX, args.DATA_DIR)
