import torch
import copy
import numpy as np
import utils.rotation_conversions as rc
from utils.pca_conversions import pca_to_matrix


def quaternion_between_vectors(v1, v2):
    assert v1.shape == v2.shape == (3,)

    # Normalize the vectors
    v1 = v1 / (torch.norm(v1) + 1e-8)
    v2 = v2 / (torch.norm(v2) + 1e-8)

    # Compute the cross product and dot product
    cross_prod = torch.cross(v1, v2, dim=-1)
    dot_prod = torch.dot(v1, v2)

    # Construct the quaternion by concatenating the scalar and vector parts
    q = torch.cat((torch.tensor([1.0 + dot_prod]), cross_prod))

    # Normalize the quaternion
    q = q / torch.norm(q)
    return q


def quaternion_between_vectors_batch(v1, v2):

    # Normalize the vectors
    v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-8)
    v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-8)

    # Compute the cross product and dot product
    cross_prod = torch.cross(v1, v2, dim=-1)
    dot_prod = (v1 * v2).sum(dim=-1)

    # Construct the quaternion by concatenating the scalar and vector parts
    q = torch.cat((1.0 + dot_prod[..., None], cross_prod), dim=-1)

    # Normalize the quaternion
    q = q / torch.norm(q, dim=-1, keepdim=True)
    return q


def rotation_to_make_this_forward_batch(rotmat):
    # rotmat: B x 3 X 3
    assert rotmat.shape[1:] == (3, 3)
    rotq = rc.matrix_to_quaternion(rotmat)  # B x 4
    B = rotmat.shape[0]

    # Where will the vector (1, 0, 0) go after applying the rotation?
    forward = rc.quaternion_apply(rotq, torch.tensor([1.0, 0, 0])[None].repeat(B, 1)).float()
    # Make it lie in the x-y plane
    forward = torch.tensor([1.0, 1, 0]).float()[None].repeat(B, 1) * forward
    forward = forward / (torch.norm(forward, dim=-1, keepdim=True) + 1e-8)

    # Compute the rotation that rotates it back to (1, 0, 0)
    ret = rc.quaternion_invert(
        quaternion_between_vectors_batch(torch.tensor([1.0, 0, 0]).float()[None].repeat(B, 1), forward)
    )
    ret = rc.quaternion_to_matrix(ret)

    return ret


def rot_trans_to_matrix(rot, trans):
    # rot: ... x 3 x 3
    # trans: ... x 3
    # returns ... x 4 x 4

    shape = list(rot.shape)[:-2]

    # use concate
    T = torch.cat([rot, trans[..., :, None]], dim=-1)  # ... x 3 x 4
    pad = torch.tensor([[0, 0, 0, 1]]).float().to(rot.device)  # 1 x 4
    pad = pad.repeat(*shape, 1, 1)  # ... x 4
    T = torch.cat([T, pad], dim=-2)  # ... x 4 x 4
    return T


def rotation_to_make_this_forward(rotmat):
    # rotmat: 3 X 3
    rotq = rc.matrix_to_quaternion(rotmat)

    # Where will the vector (1, 0, 0) go after applying the rotation?
    forward = rc.quaternion_apply(rotq, torch.tensor([1.0, 0, 0]))
    # Make it lie in the x-y plane
    forward = torch.tensor([1.0, 1, 0]).float() * forward
    forward = forward / (torch.norm(forward) + 1e-8)

    # Compute the rotation that rotates it back to (1, 0, 0)
    ret = rc.quaternion_invert(quaternion_between_vectors(torch.tensor([1.0, 0, 0]), forward))
    ret = rc.quaternion_to_matrix(ret)

    return ret


def canonicalize_trajectory(aria_traj, smpl_params, kp3d):
    """Canonicalize the motion sequence to make the first aria frame face -Y direction
    and at the origin of XY plane."""

    # transform aria to align its first frame
    tsfm_rot = rotation_to_make_this_forward(aria_traj[0, :3, :3])  # 3 x 3
    tsfm_trans = aria_traj[0, :3, 3] * torch.tensor([-1.0, -1, 0])  # 3 , z is up
    tsfm_trans = (tsfm_rot @ tsfm_trans[:, None])[:, 0]  # 3
    # tsfm_trans = torch.zeros(3)
    tsfm_T = rot_trans_to_matrix(tsfm_rot, tsfm_trans)  # 4 x 4
    can_aria_traj = tsfm_T[None] @ aria_traj  # T x 4 x 4

    # global orient is NOT defined at the origin but pelvis joint i.e. kp3d[:, 0]
    # in canonical pose which depends on shape. So there is this extra transformation
    # that we need to take care of. This makes the canonicalization a bit more complicated.

    # get body transform
    b_rot = smpl_params["global_orient"]  # T x 3 x 3
    b_transl = smpl_params["transl"]  # T x 3
    b_pelvis = kp3d[:, 0]  # T x 3
    b_T = rot_trans_to_matrix(b_rot, b_pelvis)  # T x 4 x 4

    # # get transformation to keep the relative pose of body wrt aria same
    # rel_T = b_T[0] @ torch.inverse(aria_traj[0])  # 4 x 4
    # tsfm_T = rel_T @ tsfm_T @ torch.inverse(rel_T)  # 4 x 4

    # transform body
    can_b_T = tsfm_T[None] @ b_T  # T x 4 x 4

    # get canonical smpl params
    can_b_rot = can_b_T[:, :3, :3]
    can_b_pelvis = can_b_T[:, :3, 3]
    can_b_transl = can_b_pelvis - (b_pelvis - b_transl)

    can_smpl_params = copy.deepcopy(smpl_params)
    can_smpl_params["global_orient"] = can_b_rot
    can_smpl_params["transl"] = can_b_transl

    # get canonical kp3d
    can_kp3d = (tsfm_rot[None, None] @ kp3d[..., None])[..., 0] + tsfm_trans[None, None]
    # kp3d_homo = torch.cat([kp3d, torch.ones_like(kp3d[..., :1])], dim=-1)
    # can_kp3d = (tsfm_T[None, None] @ kp3d_homo[..., None])[..., 0][..., :3]
    return can_aria_traj, can_smpl_params, can_kp3d
    return can_aria_traj, can_smpl_params, b_T, can_b_T


def get_a_canonicalized_segment(smpl_params, aria_traj, kp3d, smpl, start_idx, end_idx):

    seg_smpl_params = {k: v[start_idx : end_idx + 1].clone() for k, v in smpl_params.items()}
    seg_aria_traj = aria_traj[start_idx : end_idx + 1].clone()
    seg_kp3d = kp3d[start_idx : end_idx + 1].clone()

    seg_smpl_params["betas"] = smpl_params["betas"][:1].clone()  # We only saved one beta, not full T
    seg_aria_traj, seg_smpl_params = saved_sequence_to_full_sequence(seg_aria_traj, seg_smpl_params, smpl)

    can_aria_traj, can_smpl_params, can_kp3d = canonicalize_trajectory(seg_aria_traj, seg_smpl_params, seg_kp3d)
    # can_kp3d, can_verts, can_full_pose = evaluate_smpl(smpl, can_smpl_params)
    # smpl_output = smpl(**can_smpl_params, return_full_pose=True)
    # can_kp3d = smpl_output.joints[:, :76]
    # can_verts = smpl_output.vertices
    # can_full_pose = smpl_output.full_pose

    # body root is not at origin and depends on beta
    body_root_offset = can_kp3d[0, 0] - can_smpl_params["transl"][0]  # 3

    data_dict = {
        "start_idx": start_idx * 3,
        "end_idx": end_idx * 3,
        "can_aria_traj": can_aria_traj,
        "can_smpl_params": can_smpl_params,
        "can_kp3d": can_kp3d[:, :76],  # only body and hands
        "body_root_offset": body_root_offset,
        # debug
        # "can_full_pose": can_full_pose,
    }
    return data_dict


def saved_sequence_to_full_sequence(aria_traj, smpl_params, smpl):
    # Saved smpl params are in in 6D rotation format and body shape is not repeated for all frames.
    # Convert them to full sequence format for easier use.

    aria_traj_T = torch.eye(4)[None].repeat(aria_traj.shape[0], 1, 1).to(aria_traj.device)
    aria_traj_T[:, :3, :3] = rc.rotation_6d_to_matrix(aria_traj[:, :6])
    aria_traj_T[:, :3, 3] = aria_traj[:, 6:9]

    ret_smpl_params = {}
    ret_smpl_params["global_orient"] = rc.rotation_6d_to_matrix(smpl_params["global_orient"])
    ret_smpl_params["body_pose"] = rc.rotation_6d_to_matrix(smpl_params["body_pose"])
    ret_smpl_params["betas"] = smpl_params["betas"].repeat(aria_traj.shape[0], 1)
    ret_smpl_params["transl"] = smpl_params["transl"].clone()
    if "left_hand_pose" in smpl_params:
        ret_smpl_params["left_hand_pose"] = pca_to_matrix(smpl_params["left_hand_pose"], smpl.left_hand_components)
        ret_smpl_params["right_hand_pose"] = pca_to_matrix(smpl_params["right_hand_pose"], smpl.right_hand_components)

    return aria_traj_T, ret_smpl_params
