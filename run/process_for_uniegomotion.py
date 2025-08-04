import argparse
import copy
import json
import os

import joblib
import torch
from loguru import logger
from tqdm.auto import tqdm

import utils.rotation_conversions as rc
from dataset.egoego_utils import (
    determine_floor_height_and_contacts,
)
from dataset.egoexo4d_take_dataset import EgoExo4D_Take_Dataset
from dataset.smpl_utils import get_smpl, evaluate_smpl
from dataset.egoexo4d_utils import ego_extri_to_egoego_head_traj, smooth_sequence
from utils.pca_conversions import matrix_to_pca, pca_to_matrix
from utils.rotation_conversions import matrix_to_rotation_6d
from utils.torch_utils import to_device, to_numpy, to_tensor

MIN_SEGMENT_LEN = 20  # 2 seconds at 10fps
MAX_VEL_THRESH = 5  # m/s
MAX_ACC_THRESH = 40  # m/s^2
MIN_VEL_STATIC_THRESH = 0.1  # m/s
ARIA_CALIB_ERROR_THRESH = 0.3  # m
MIN_NUM_CAMS = 3


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/vision/u/chpatel/data/egoexo4d_ee4d_motion/")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val"])

    return parser.parse_args()


def smooth_smpl_sequence(smpl_params):
    # smooth the sequence
    def smooth_rotmat(x):
        x = rc.matrix_to_quaternion(x)
        x = rc.standardize_quaternion(smooth_sequence(x))
        x = x / torch.norm(x, dim=-1, keepdim=True)
        x = rc.quaternion_to_matrix(x)
        return x

    smpl_params["global_orient"] = smooth_rotmat(smpl_params["global_orient"])
    smpl_params["body_pose"] = smooth_rotmat(smpl_params["body_pose"])
    if "left_hand_pose" in smpl_params:
        smpl_params["left_hand_pose"] = smooth_rotmat(smpl_params["left_hand_pose"])
        smpl_params["right_hand_pose"] = smooth_rotmat(smpl_params["right_hand_pose"])
    smpl_params["transl"] = smooth_sequence(smpl_params["transl"])

    return smpl_params


def get_segments(frame_idxs):
    # find segments
    if len(frame_idxs) == 0:
        return []

    max_frame_num = frame_idxs[-1]
    left_idx = None
    segments = []
    for idx in range(0, max_frame_num + 1, 3):
        if idx in frame_idxs:
            if left_idx is None:  # found first frame of the segment
                left_idx = idx
            continue

        if left_idx is not None:  # continuing segment broken
            segments.append((left_idx, idx - 3))
            left_idx = None
    if left_idx is not None:  # last segment
        segments.append((left_idx, max_frame_num))

    # discard short segments
    new_segments = []
    for start_idx, end_idx in segments:
        if (end_idx - start_idx + 3) // 3 > MIN_SEGMENT_LEN:  # more than 2 seconds
            new_segments.append((start_idx, end_idx))
    segments = new_segments

    return segments


def find_good_segments(dataset, take_name, smpl, bdata, aria_traj):
    data_dir = dataset.data_dir
    frame_idxs = sorted(list(bdata.keys()))
    total_video_frames = max(len(frame_idxs), int(dataset.metadata[take_name]["duration_sec"] * 10))
    # logger.info(f"Available {100 * len(frame_idxs) / total_video_frames} percent frames")

    # Find frames with more than one visible camera
    bboxes = joblib.load(f"{data_dir}/ee4d_motion/{take_name}/bboxes_clean.pkl")["bboxes"]
    new_frame_idxs = []
    for idx in frame_idxs:
        num_vis_cams = 0
        for cam, cam_bboxes in bboxes.items():
            if idx in cam_bboxes:
                num_vis_cams += 1

        if num_vis_cams >= MIN_NUM_CAMS:
            new_frame_idxs.append(idx)
    # logger.info(f"Retained {100 * len(new_frame_idxs) / len(frame_idxs)} percent of frames")
    frame_idxs = sorted(new_frame_idxs)

    if len(frame_idxs) == 0:
        logger.info(f"{take_name}. No frames with enough visible camera. Skipping.")
        return None

    segments = get_segments(frame_idxs)

    # logger.warning("Not cleaning.")
    # return segments

    if len(segments) == 0:
        logger.info(f"{take_name}. No segments found. Skipping.")
        return None

    # process each segment
    # logger.info(f"Segments: {segments}")
    all_bad_frames = []
    for start_idx, end_idx in segments:

        seg_bdata = [bdata[fid] for fid in range(start_idx, end_idx + 3, 3)]
        seg_bdata = torch.utils.data.default_collate(seg_bdata)  # make everything batched
        smpl_params = copy.deepcopy(seg_bdata["final_smpl_params"])

        smpl_params["global_orient"] = rc.rotation_6d_to_matrix(smpl_params["global_orient"])
        smpl_params["body_pose"] = rc.rotation_6d_to_matrix(smpl_params["body_pose"])
        if "left_hand_pose" in smpl_params:
            smpl_params["right_hand_pose"] = pca_to_matrix(smpl_params["right_hand_pose"], smpl.right_hand_components)
            smpl_params["left_hand_pose"] = pca_to_matrix(smpl_params["left_hand_pose"], smpl.left_hand_components)

        kp3d, verts, full_pose = evaluate_smpl(smpl, smpl_params)

        # Find bad frames by high velocity and acceleration
        # Get smooth velocity and acceleration data
        kp3d_smooth = smooth_sequence(kp3d[:, :22])
        kp3d_vel = (kp3d_smooth[1:] - kp3d_smooth[:-1]) * 10
        kp3d_acc = (kp3d_vel[1:] - kp3d_vel[:-1]) * 10
        kp3d_vel = torch.cat([kp3d_vel, kp3d_vel[-1:]], dim=0)
        kp3d_acc = torch.cat([kp3d_acc[:1], kp3d_acc, kp3d_acc[-1:]], dim=0)
        # Filter out high velocity and acceleration frames  T x J
        high_vel_idx = kp3d_vel.square().sum(-1).sqrt() > MAX_VEL_THRESH  # 5 m/s
        high_acc_idx = kp3d_acc.square().sum(-1).sqrt() > MAX_ACC_THRESH  # 40 m/s^2

        # Find static poses without any motion
        # Double smooth the data
        kp3d_smooth = smooth_sequence(smooth_sequence(kp3d[:, :22]))
        kp3d_vel = (kp3d_smooth[1:] - kp3d_smooth[:-1]) * 10
        kp3d_vel = torch.cat([kp3d_vel, kp3d_vel[-1:]], dim=0)
        kp3d_vel = kp3d_vel.square().sum(-1).sqrt().max(-1).values  # T, max joint velocity at each frame
        # 1d maxpool over a window of 20 frames
        kp3d_vel = torch.nn.functional.max_pool1d(kp3d_vel[None], kernel_size=15, stride=1, padding=7)[0]
        assert kp3d_vel.shape[0] == kp3d.shape[0]
        static_idx = kp3d_vel < MIN_VEL_STATIC_THRESH  # 0.1 m/s

        # Bad aria SLAM
        leye_idx = 23
        aria_transl = torch.stack([aria_traj[fid][:3, 3] for fid in range(start_idx, end_idx + 3, 3)])
        eye_to_aria = kp3d[:, leye_idx] - aria_transl
        eye_to_aria = eye_to_aria.square().sum(-1).sqrt()  # T
        bad_aria_calib_idx = eye_to_aria > ARIA_CALIB_ERROR_THRESH  # 30 cm

        # Find bad frames
        bad_frames = (high_vel_idx | high_acc_idx).any(-1) | static_idx | bad_aria_calib_idx
        bad_frames = [i * 3 + start_idx for i, b in enumerate(bad_frames) if b]
        all_bad_frames.extend(bad_frames)

    good_frames = list(set(frame_idxs) - set(all_bad_frames))
    segments = get_segments(good_frames)
    remaining_frames = sum([(end_idx - start_idx + 3) // 3 for start_idx, end_idx in segments])
    # logger.info(f"{take_name}. Segments after filtering: {segments}.")
    logger.info(f"{take_name}. Retained {100 * remaining_frames / total_video_frames} percent of frames")

    return segments


def main(args, dataset, take_name):
    device = torch.device("cpu")
    data_dir = args.data_dir

    # Load SMPL data
    bdata_path = f"{data_dir}/ee4d_motion/{take_name}/multiview_optim_2.pkl"
    if not os.path.exists(bdata_path):
        logger.info(f"{take_name}. Multiview optim does not exist. Skipping.")
        return None
    bdata = joblib.load(bdata_path)
    bdata = to_tensor(bdata)

    # Load SMPL model
    smpl = get_smpl().to(device)

    # Load ARIA trajectory
    ego_extri = dataset.get_ego_extri_directly(take_name)
    aria_traj = ego_extri_to_egoego_head_traj(ego_extri)
    aria_traj = to_tensor(aria_traj)
    aria_traj = to_device(aria_traj, device)

    # Find good segments
    segments = find_good_segments(dataset, take_name, smpl, bdata, aria_traj)
    if segments is None:
        logger.info(f"{take_name}. No good segments found. Skipping.")
        return None
    ret = {}
    for sidx, (start_idx, end_idx) in enumerate(segments):
        frame_idxs = list(range(start_idx, end_idx + 3, 3))
        seg_aria_traj = torch.stack([aria_traj[fid] for fid in frame_idxs])

        seg_bdata = [bdata[fid] for fid in range(start_idx, end_idx + 3, 3)]
        seg_bdata = torch.utils.data.default_collate(seg_bdata)  # make everything batched
        smpl_params = copy.deepcopy(seg_bdata["final_smpl_params"])

        smpl_params["global_orient"] = rc.rotation_6d_to_matrix(smpl_params["global_orient"])
        smpl_params["body_pose"] = rc.rotation_6d_to_matrix(smpl_params["body_pose"])
        if "left_hand_pose" in smpl_params:
            smpl_params["right_hand_pose"] = pca_to_matrix(smpl_params["right_hand_pose"], smpl.right_hand_components)
            smpl_params["left_hand_pose"] = pca_to_matrix(smpl_params["left_hand_pose"], smpl.left_hand_components)

        # smooth SMPL sequence
        smpl_params = smooth_smpl_sequence(smpl_params)

        # Evaluate SMPL
        kp3d, verts, full_pose = evaluate_smpl(smpl, smpl_params)

        # Neutral body
        # leye_idx = 23
        # aria_from_leye = seg_aria_traj[:, :3, 3] - kp3d[:, leye_idx]
        # smpl_params["betas"] = torch.zeros_like(smpl_params["betas"])
        # kp3d, verts, full_pose = evaluate_smpl(smpl, smpl_params)
        # seg_aria_traj[:, :3, 3] = kp3d[:, leye_idx] + aria_from_leye

        # Determine floor height and contacts
        floor_height, contacts, discard_seq = determine_floor_height_and_contacts(to_numpy(kp3d), fps=10)

        # Make Things Small
        smpl_params["global_orient"] = matrix_to_rotation_6d(smpl_params["global_orient"])  # T x 6
        smpl_params["body_pose"] = matrix_to_rotation_6d(smpl_params["body_pose"])  # T x (21*6)
        assert torch.allclose(smpl_params["betas"], smpl_params["betas"][0:1])
        smpl_params["betas"] = smpl_params["betas"][:1]  # 1 x 10 because betas are same
        if "left_hand_pose" in smpl_params:
            smpl_params["left_hand_pose"] = matrix_to_pca(smpl_params["left_hand_pose"], smpl.left_hand_components)
            smpl_params["right_hand_pose"] = matrix_to_pca(smpl_params["right_hand_pose"], smpl.right_hand_components)

        seg_aria_traj_rot6d = matrix_to_rotation_6d(seg_aria_traj[:, :3, :3])  # T x 6
        seg_aria_traj = torch.cat([seg_aria_traj_rot6d, seg_aria_traj[:, :3, 3]], dim=1)  # T x (6 + 3)

        # body root is not at origin and depends on beta
        body_root_offset = kp3d[0, 0] - smpl_params["transl"][0]  # 3

        data_dict = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "aria_traj": seg_aria_traj,
            "smpl_params": smpl_params,
            "kp3d": kp3d[:, :76],  # only body and hands
            "body_root_offset": body_root_offset,
            "floor_height": floor_height,
            "num_frames": seg_aria_traj.shape[0],
        }
        ret[f"{take_name}___{start_idx}___{end_idx}"] = data_dict

    logger.info(f"{take_name}. Done.")
    return ret


def save_segment_data():
    args = get_args()

    with open(f"{args.data_dir}/annotations/splits.json", "r") as f:
        splits = json.load(f)["split_to_take_uids"]

    dataset = EgoExo4D_Take_Dataset(args.data_dir, lowres=True)
    out_dict = {}
    out_file = f"{args.data_dir}/uniegomotion/ee_{args.split}.pt"

    if os.path.exists(out_file) and not args.overwrite:
        logger.info(f"{out_file} exists. Skipping.")
        return

    for take_uid in tqdm(splits[args.split]):
        take_name = dataset.uid2name[take_uid]
        # take_name = "cmu_bike02_4"
        # if take_name not in ["utokyo_soccer_8000_43_6"]:
        #     continue

        if take_name not in dataset.metadata:
            continue

        if dataset.metadata[take_name]["parent_task_name"] == "Rock Climbing":
            continue

        ret = main(args, dataset, take_name)
        if ret is not None:
            out_dict.update(ret)

        # if len(out_dict) > 0:
        #     break

    torch.save(out_dict, out_file)


if __name__ == "__main__":
    # main()
    save_segment_data()
