import copy

import numpy as np
import torch

import utils.rotation_conversions as rc
from dataset.canonicalization import rot_trans_to_matrix, rotation_to_make_this_forward_batch
from dataset.egoego_utils import local2global_pose, mat_ik_torch
from utils.pca_conversions import matrix_to_pca, pca_to_matrix


"""
-----------------------------------------
Motion representations explanations
-----------------------------------------
v1 = global rotation and translation of each joint of each frame. Does not include betas. Used in Egoego.
v1_beta = v1 with betas appended in the motion representation. Global Motion Repre. in the paper.
v4 = Head trajectory encoded as residuals from previous frame.
     Other joints are encoded as local transformation wrt 'floor projected head trajectory'.
     Does not include betas.
v4_beta = v4 with betas appended in the motion representation. Our motion representation in the paper.
v5_beta = Same as v4_beta but pelvis centric instead of head centric. Pelvis-centric Repre. in the paper.

See the paper and supplementary material for more details.
-----------------------------------------
"""


def foot_detect(kp3d):
    # kp3d: T x 22 x 3
    fid_l, fid_r = 10, 11
    velfactor, heightfactor = 0.02, 0.10

    vel_l = kp3d[1:, fid_l] - kp3d[:-1, fid_l]
    vel_l = vel_l.square().sum(-1).sqrt()
    vel_l = (vel_l < velfactor).float()
    vel_l = torch.cat([vel_l[:1], vel_l], dim=0)

    vel_r = kp3d[1:, fid_r] - kp3d[:-1, fid_r]
    vel_r = vel_r.square().sum(-1).sqrt()
    vel_r = (vel_r < velfactor).float()
    vel_r = torch.cat([vel_r[:1], vel_r], dim=0)

    height_l = (kp3d[:, fid_l, 2] < heightfactor).float()
    height_r = (kp3d[:, fid_r, 2] < heightfactor).float()

    feet_l = vel_l * height_l
    feet_r = vel_r * height_r
    return feet_l, feet_r


def saved_sequence_to_full_sequence(aria_traj, smpl_params, smpl):
    # Saved smpl params are in in 6D rotation format and body shape is not repeated for all frames.
    # Convert them to full sequence format for easier use.

    if aria_traj.ndim == 3:
        # maybe already full sequence. check it.
        T = aria_traj.shape[0]
        assert aria_traj.shape == (T, 4, 4)
        assert smpl_params["global_orient"].shape == (T, 3, 3)
        assert smpl_params["body_pose"].shape == (T, 21, 3, 3)
        assert smpl_params["transl"].shape == (T, 3)
        assert smpl_params["betas"].shape == (T, 10)
        if "left_hand_pose" in smpl_params:
            assert smpl_params["left_hand_pose"].shape == (T, 15, 3, 3)
            assert smpl_params["right_hand_pose"].shape == (T, 15, 3, 3)
        return aria_traj.clone(), copy.deepcopy(smpl_params)

    aria_traj_T = torch.eye(4)[None].repeat(aria_traj.shape[0], 1, 1).to(aria_traj.device)
    aria_traj_T[:, :3, :3] = rc.rotation_6d_to_matrix(aria_traj[:, :6])
    aria_traj_T[:, :3, 3] = aria_traj[:, 6:9]

    smpl_params = copy.deepcopy(smpl_params)
    smpl_params["global_orient"] = rc.rotation_6d_to_matrix(smpl_params["global_orient"])
    smpl_params["body_pose"] = rc.rotation_6d_to_matrix(smpl_params["body_pose"])
    smpl_params["betas"] = smpl_params["betas"].repeat(aria_traj.shape[0], 1)
    if "left_hand_pose" in smpl_params:
        smpl_params["left_hand_pose"] = pca_to_matrix(smpl_params["left_hand_pose"], smpl.left_hand_components)
        smpl_params["right_hand_pose"] = pca_to_matrix(smpl_params["right_hand_pose"], smpl.right_hand_components)

    return aria_traj_T, smpl_params


def common_saved_to_repre(aria_traj, smpl_params, kp3d, floor_height, smpl):
    # Some common operations for all representations. Avoids code duplication.
    # global_T is body joint transformations in global frame, not wrt parent joint.

    # left_hand_pca = smpl_params["left_hand_pose"].clone()
    # right_hand_pca = smpl_params["right_hand_pose"].clone()
    aria_traj_T, smpl_params = saved_sequence_to_full_sequence(aria_traj, smpl_params, smpl)
    left_hand_pca = matrix_to_pca(smpl_params["left_hand_pose"], smpl.left_hand_components)
    right_hand_pca = matrix_to_pca(smpl_params["right_hand_pose"], smpl.right_hand_components)

    if floor_height == 0:
        # print(f"Floor height is not available: {floor_height}")
        floor_height = kp3d[:, [10, 11], 2].mean().item() - 0.02

    # apply offset for floor height
    offset = torch.tensor([0, 0, floor_height]).float().to(aria_traj.device)
    aria_traj_T[:, :3, 3] = aria_traj_T[:, :3, 3] - offset
    kp3d = kp3d - offset
    smpl_params["transl"] = smpl_params["transl"] - offset
    del offset

    feet_l, feet_r = foot_detect(kp3d)  # T

    # Get global transformations for each joint
    T = aria_traj_T.shape[0]
    full_pose = torch.cat(
        [
            smpl_params["global_orient"][:, None],
            smpl_params["body_pose"],
            # torch.eye(3)[None, None].repeat(T, 1, 1, 1).to(aria_traj.device),  # jaw
            # torch.eye(3)[None, None].repeat(T, 1, 1, 1).to(aria_traj.device),  # leye
            # torch.eye(3)[None, None].repeat(T, 1, 1, 1).to(aria_traj.device),  # reye
            # smpl_params["left_hand_pose"],
            # smpl_params["right_hand_pose"],
        ],
        dim=1,
    )  # T x 22 x 3 x 3
    global_rotmat = local2global_pose(full_pose, list(smpl.parents)[:22])  # T x 22 x 3 x 3
    global_trans = kp3d[:, :22]
    global_T = rot_trans_to_matrix(global_rotmat, global_trans)  # T x 22 x 4 x 4

    return global_T, feet_l, feet_r, aria_traj_T, smpl_params, left_hand_pca, right_hand_pca


def saved_sequence_to_repre_v1(aria_traj, smpl_params, kp3d, floor_height, smpl):
    # Explanation of representations is at the top.

    global_T, feet_l, feet_r, aria_traj_T, smpl_params, left_hand_pca, right_hand_pca = common_saved_to_repre(
        aria_traj, smpl_params, kp3d, floor_height, smpl
    )

    global_rot6d = rc.matrix_to_rotation_6d(global_T[:, :22, :3, :3])  # T x 22 x 6
    global_rottrans = torch.cat([global_rot6d, global_T[:, :22, :3, 3]], dim=2)  # T x 22 x 9
    T = aria_traj_T.shape[0]
    x = torch.cat(
        [
            global_rottrans.view(T, 198),  # T x 198
            left_hand_pca,  # T x 12
            right_hand_pca,  # T x 12
            feet_l.view(T, 1),  # T x 1
            feet_r.view(T, 1),  # T x 1
        ],
        dim=1,
    )  # T x (198+12+12+1+1)
    x = x.view(T, 224)

    aria_traj_rot6d = rc.matrix_to_rotation_6d(aria_traj_T[:, :3, :3])  # T x 6
    aria_traj_trans = aria_traj_T[:, :3, 3]  # T x 3
    aria_traj_repre = torch.cat([aria_traj_rot6d, aria_traj_trans], dim=1)  # T x 9

    return x, aria_traj_repre


def repre_to_full_sequence_v1(x, aria_traj_repre, smpl, betas, body_root_offset):
    # Explanation of representations is at the top.
    T = aria_traj_repre.shape[0] if aria_traj_repre is not None else x.shape[0]
    if aria_traj_repre is not None:
        aria_traj_rotmat = rc.rotation_6d_to_matrix(aria_traj_repre[:, :6])
        aria_traj_trans = aria_traj_repre[:, 6:9]
        aria_traj_T = rot_trans_to_matrix(aria_traj_rotmat, aria_traj_trans)
    else:
        aria_traj_T = None

    if x is None:
        return aria_traj_T, None, None

    global_rottrans = x[:, :198].view(T, 22, 9)  # T x 22 x 9
    left_hand_pca = x[:, 198:210]  # T x 12
    right_hand_pca = x[:, 210:222]  # T x 12

    global_rotmat = rc.rotation_6d_to_matrix(global_rottrans[:, :, :6])  # T x 22 x 3 x 3
    global_trans = global_rottrans[:, :, 6:9]  # T x 22 x 3
    local_rotmat = mat_ik_torch(global_rotmat, list(smpl.parents)[:22])  # T x 22 x 3 x 3

    smpl_params = {}
    smpl_params["global_orient"] = local_rotmat[:, 0]  # T x 3 x 3
    smpl_params["body_pose"] = local_rotmat[:, 1:]  # T x 21 x 3 x 3
    smpl_params["betas"] = betas.view(-1, 10).repeat(T, 1)
    smpl_params["left_hand_pose"] = pca_to_matrix(left_hand_pca, smpl.left_hand_components)
    smpl_params["right_hand_pose"] = pca_to_matrix(right_hand_pca, smpl.right_hand_components)

    if body_root_offset is None:
        body_root_offset = smpl.forward(betas=betas.view(-1, 10)).joints[0, 0]  # 3
    smpl_params["transl"] = global_trans[:, 0] - body_root_offset[None]  # T x 3

    j3d = global_trans

    return aria_traj_T, smpl_params, j3d


def saved_sequence_to_repre_v1_beta(aria_traj, smpl_params, kp3d, floor_height, smpl):
    # Explanation of representations is at the top.
    x, aria_traj_repre = saved_sequence_to_repre_v1(aria_traj, smpl_params, kp3d, floor_height, smpl)
    # append beta to x
    betas = smpl_params["betas"]  # T x 10
    assert betas.shape[-1] == 10
    x = torch.cat([x, betas], dim=1)  # T x 234
    return x, aria_traj_repre


def repre_to_full_sequence_v1_beta(x, aria_traj_repre, smpl, betas, body_root_offset):
    # Explanation of representations is at the top.
    # Modify gt betas with the mean of the predicted betas
    x, x_betas = x[:, :-10], x[:, -10:]
    x_betas = x_betas.mean(0)[None]  # 1 x 10
    body_root_offset = None
    return repre_to_full_sequence_v1(x, aria_traj_repre, smpl, x_betas, body_root_offset)


def saved_sequence_to_repre_v4(aria_traj, smpl_params, kp3d, floor_height, smpl, canon_root_idx=23):  # 23 is leye
    # Explanation of representations is at the top.

    global_T, feet_l, feet_r, aria_traj_T, smpl_params, left_hand_pca, right_hand_pca = common_saved_to_repre(
        aria_traj, smpl_params, kp3d, floor_height, smpl
    )
    T = aria_traj_T.shape[0]

    # Canonicalization of leye
    if canon_root_idx >= global_T.shape[1]:
        leye_T = aria_traj_T.clone()
    else:
        leye_T = global_T[:, canon_root_idx]  # T x 4 x 4
    tsfm_rot = rotation_to_make_this_forward_batch(leye_T[:, :3, :3])  # T x 3 x 3
    tsfm_trans = leye_T[:, :3, 3] * torch.tensor([-1.0, -1, 0])  # T x 3
    tsfm_trans = (tsfm_rot @ tsfm_trans[..., None])[..., 0]  # T x 3
    tsfm_T = rot_trans_to_matrix(tsfm_rot, tsfm_trans)  # T x 4 x 4
    local_T = tsfm_T[:, None] @ global_T  # T x 22 x 4 x 4

    # Canonicalization as delta
    invtsfm_T = tsfm_T.inverse()
    delta_invtsfm_T = invtsfm_T[:-1].inverse() @ invtsfm_T[1:]  # (T-1) x 4 x 4
    delta_invtsfm_T = torch.cat([invtsfm_T[:1], delta_invtsfm_T], dim=0)  # T x 4 x 4

    # representation
    local_rottrans = torch.cat(
        [rc.matrix_to_rotation_6d(local_T[:, :22, :3, :3]), local_T[:, :22, :3, 3]], dim=2
    )  # T x nj x 9
    delta_invtsfm_rottrans = torch.cat(
        [rc.matrix_to_rotation_6d(delta_invtsfm_T[:, :3, :3]), delta_invtsfm_T[:, :3, 3]], dim=1
    )  # T x 9

    x = torch.cat(
        [
            # local_rottrans.view(T, 225),  # T x 225
            local_rottrans.view(T, 198),  # T x 198
            delta_invtsfm_rottrans,  # T x 9
            left_hand_pca,  # T x 12
            right_hand_pca,  # T x 12
            feet_l.view(T, 1),  # T x 1
            feet_r.view(T, 1),  # T x 1
        ],
        dim=1,
    )
    x = x.view(T, 233)

    # Aria trajectory canonicalization
    aria_tsfm_rot = rotation_to_make_this_forward_batch(aria_traj_T[:, :3, :3])  # T x 3 x 3
    aria_tsfm_trans = aria_traj_T[:, :3, 3] * torch.tensor([-1.0, -1, 0])  # T x 3
    aria_tsfm_trans = (aria_tsfm_rot @ aria_tsfm_trans[..., None])[..., 0]  # T x 3
    aria_tsfm_T = rot_trans_to_matrix(aria_tsfm_rot, aria_tsfm_trans)  # T x 4 x 4
    aria_local_T = aria_tsfm_T @ aria_traj_T  # T x 4 x 4

    # Canonical frame as delta
    aria_invtsfm_T = aria_tsfm_T.inverse()
    delta_aria_invtsfm_T = aria_invtsfm_T[:-1].inverse() @ aria_invtsfm_T[1:]  # (T-1) x 4 x 4
    delta_aria_invtsfm_T = torch.cat([aria_invtsfm_T[:1], delta_aria_invtsfm_T], dim=0)  # T x 4 x 4

    # representation
    aria_local_rottrans = torch.cat(
        [rc.matrix_to_rotation_6d(aria_local_T[:, :3, :3]), aria_local_T[:, :3, 3]], dim=1
    )  # T x 9
    delta_aria_invtsfm_rottrans = torch.cat(
        [rc.matrix_to_rotation_6d(delta_aria_invtsfm_T[:, :3, :3]), delta_aria_invtsfm_T[:, :3, 3]], dim=1
    )  # T x 9
    aria_traj_repre = torch.cat([aria_local_rottrans, delta_aria_invtsfm_rottrans], dim=1)  # T x 18

    return x, aria_traj_repre


def repre_to_full_sequence_v4(x, aria_traj_repre, smpl, betas, body_root_offset):
    # Explanation of representations is at the top.
    T = aria_traj_repre.shape[0] if aria_traj_repre is not None else x.shape[0]

    if aria_traj_repre is not None:
        # Get aria trajectory
        aria_local_T = rot_trans_to_matrix(rc.rotation_6d_to_matrix(aria_traj_repre[:, :6]), aria_traj_repre[:, 6:9])
        delta_aria_invtsfm_T = rot_trans_to_matrix(
            rc.rotation_6d_to_matrix(aria_traj_repre[:, 9:15]), aria_traj_repre[:, 15:18]
        )
        aria_invtsfm_T = [delta_aria_invtsfm_T[0]]
        for i in range(1, T):
            aria_invtsfm_T.append(aria_invtsfm_T[-1] @ delta_aria_invtsfm_T[i])
        aria_invtsfm_T = torch.stack(aria_invtsfm_T, dim=0)  # T x 4 x 4
        aria_traj_T = aria_invtsfm_T @ aria_local_T
    else:
        aria_traj_T = None

    if x is None:
        return aria_traj_T, None, None

    # Get body
    local_rottrans = x[:, :198].view(T, 22, 9)  # T x 22 x 9
    local_T = rot_trans_to_matrix(
        rc.rotation_6d_to_matrix(local_rottrans[:, :, :6]), local_rottrans[:, :, 6:9]
    )  # T x 22 x 4 x 4
    delta_invtsfm_rottrans = x[:, 198 : 198 + 9]  # T x 9
    delta_invtsfm_T = rot_trans_to_matrix(
        rc.rotation_6d_to_matrix(delta_invtsfm_rottrans[:, :6]), delta_invtsfm_rottrans[:, 6:9]
    )  # T x 4 x 4
    invtsfm_T = [delta_invtsfm_T[0]]
    for i in range(1, T):
        invtsfm_T.append(invtsfm_T[-1] @ delta_invtsfm_T[i])
    invtsfm_T = torch.stack(invtsfm_T, dim=0)  # T x 4 x 4
    global_T = invtsfm_T[:, None] @ local_T  # T x 22 x 4 x 4

    global_trans = global_T[:, :, :3, 3]  # T x 22 x 3
    local_rotmat = mat_ik_torch(global_T[:, :, :3, :3], list(smpl.parents)[: global_T.shape[1]])  # T x 22 x 3 x 3
    left_hand_pca = x[:, 198 + 9 : 198 + 9 + 12]  # T x 12
    right_hand_pca = x[:, 198 + 9 + 12 : 198 + 9 + 24]  # T x 12

    smpl_params = {}
    smpl_params["global_orient"] = local_rotmat[:, 0]  # T x 3 x 3
    smpl_params["body_pose"] = local_rotmat[:, 1:22]  # T x 21 x 3 x 3
    if local_rotmat.shape[1] > 22:
        smpl_params["jaw_pose"] = local_rotmat[:, 22]
        smpl_params["leye_pose"] = local_rotmat[:, 23]
        smpl_params["reye_pose"] = local_rotmat[:, 24]
    smpl_params["betas"] = betas.view(-1, 10).repeat(T, 1)
    smpl_params["left_hand_pose"] = pca_to_matrix(left_hand_pca, smpl.left_hand_components)
    smpl_params["right_hand_pose"] = pca_to_matrix(right_hand_pca, smpl.right_hand_components)

    if body_root_offset is None:
        body_root_offset = smpl.forward(betas=betas.view(-1, 10)).joints[0, 0]  # 3
    smpl_params["transl"] = global_trans[:, 0] - body_root_offset[None]  # T x 3

    j3d = global_trans
    return aria_traj_T, smpl_params, j3d


def saved_sequence_to_repre_v4_beta(aria_traj, smpl_params, kp3d, floor_height, smpl):
    # Explanation of representations is at the top.
    x, aria_traj_repre = saved_sequence_to_repre_v4(aria_traj, smpl_params, kp3d, floor_height, smpl)
    # append beta to x
    betas = smpl_params["betas"]  # T x 10
    assert betas.shape[-1] == 10
    x = torch.cat([x, betas], dim=1)  # T x 234
    return x, aria_traj_repre


def repre_to_full_sequence_v4_beta(x, aria_traj_repre, smpl, betas, body_root_offset):
    # Explanation of representations is at the top.
    # Modify gt betas with the mean of the predicted betas
    x, x_betas = x[:, :-10], x[:, -10:]
    x_betas = x_betas.mean(0)[None]  # 1 x 10
    body_root_offset = None
    return repre_to_full_sequence_v4(x, aria_traj_repre, smpl, x_betas, body_root_offset)


def saved_sequence_to_repre_v5_beta(aria_traj, smpl_params, kp3d, floor_height, smpl):
    # Explanation of representations is at the top.
    x, aria_traj_repre = saved_sequence_to_repre_v4(aria_traj, smpl_params, kp3d, floor_height, smpl, canon_root_idx=0)
    # append beta to x
    betas = smpl_params["betas"]  # T x 10
    assert betas.shape[-1] == 10
    x = torch.cat([x, betas], dim=1)  # T x 234
    return x, aria_traj_repre


def repre_to_full_sequence_v5_beta(x, aria_traj_repre, smpl, betas, body_root_offset):
    # Explanation of representations is at the top.
    # Modify gt betas with the mean of the predicted betas
    x, x_betas = x[:, :-10], x[:, -10:]
    x_betas = x_betas.mean(0)[None]  # 1 x 10
    body_root_offset = None
    return repre_to_full_sequence_v4(x, aria_traj_repre, smpl, x_betas, body_root_offset)


def saved_sequence_to_repre(repre_type, *args, **kwargs):
    # Explanation of representations is at the top.
    if repre_type == "v1":
        return saved_sequence_to_repre_v1(*args, **kwargs)
    if repre_type == "v1_beta":
        return saved_sequence_to_repre_v1_beta(*args, **kwargs)
    if repre_type == "v4":
        return saved_sequence_to_repre_v4(*args, **kwargs)
    if repre_type == "v4_beta":
        return saved_sequence_to_repre_v4_beta(*args, **kwargs)
    if repre_type == "v5_beta":
        return saved_sequence_to_repre_v5_beta(*args, **kwargs)
    raise ValueError(f"Unknown repre_type: {repre_type}")


def repre_to_full_sequence(repre_type, *args, **kwargs):
    # Explanation of representations is at the top.
    if repre_type == "v1":
        return repre_to_full_sequence_v1(*args, **kwargs)
    if repre_type == "v1_beta":
        return repre_to_full_sequence_v1_beta(*args, **kwargs)
    if repre_type == "v4":
        return repre_to_full_sequence_v4(*args, **kwargs)
    if repre_type == "v4_beta":
        return repre_to_full_sequence_v4_beta(*args, **kwargs)
    if repre_type == "v5_beta":
        return repre_to_full_sequence_v5_beta(*args, **kwargs)
    raise ValueError(f"Unknown repre_type: {repre_type}")


if __name__ == "__main__":
    pass

    import IPython

    IPython.embed()
