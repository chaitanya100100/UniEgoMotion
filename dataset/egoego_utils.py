# Thanks to https://github.com/davrempe/humor/blob/main/humor/scripts/process_amass_data.py
# and
# https://github.com/lijiaman/egoego_release/blob/main/utils/data_utils/process_amass_dataset.py

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import DBSCAN

import utils.rotation_conversions as transforms


SMPL_JOINTS = {
    "hips": 0,
    "leftUpLeg": 1,
    "rightUpLeg": 2,
    "spine": 3,
    "leftLeg": 4,
    "rightLeg": 5,
    "spine1": 6,
    "leftFoot": 7,
    "rightFoot": 8,
    "spine2": 9,
    "leftToeBase": 10,
    "rightToeBase": 11,
    "neck": 12,
    "leftShoulder": 13,
    "rightShoulder": 14,
    "head": 15,
    "leftArm": 16,
    "rightArm": 17,
    "leftForeArm": 18,
    "rightForeArm": 19,
    "leftHand": 20,
    "rightHand": 21,
}

DISCARD_TERRAIN_SEQUENCES = True  # throw away sequences where the person steps onto objects (determined by a heuristic)

DISCARD_SHORTER_THAN = 1.0  # seconds

# optional viz during processing
VIZ_PLOTS = False
VIZ_SEQ = False

OUT_FPS = 10
# OUT_FPS = 20

# for determining floor height
FLOOR_VEL_THRESH = 0.005 * (30 / OUT_FPS)
FLOOR_HEIGHT_OFFSET = 0.01
# for determining contacts
CONTACT_VEL_THRESH = 0.005 * (30 / OUT_FPS)
CONTACT_TOE_HEIGHT_THRESH = 0.04
CONTACT_ANKLE_HEIGHT_THRESH = 0.08
# for determining terrain interaction
TERRAIN_HEIGHT_THRESH = 0.04  # if static toe is above this height
ROOT_HEIGHT_THRESH = 0.04  # if maximum "static" root height is more than this + root_floor_height
CLUSTER_SIZE_THRESH = 0.25  # if cluster has more than this faction of fps (30 for 120 fps)


def local2global_pose(local_pose, kintree):
    # Thanks to EgoEgo
    # local_pose: ... X J X 3 X 3
    global_pose = local_pose.clone()
    for jId in range(len(kintree)):
        parent_id = kintree[jId]
        if parent_id >= 0:
            global_pose[..., jId, :, :] = torch.matmul(global_pose[..., parent_id, :, :], global_pose[..., jId, :, :])
    return global_pose


def transform_vec(v, q, trans="root"):
    if trans == "root":
        rot = transforms.quaternion_to_matrix(q)  # 3 X 3
    elif trans == "heading":
        hq = q.clone()
        hq[1] = 0.0
        hq[2] = 0.0
        hq /= torch.linalg.norm(hq)
        rot = transforms.quaternion_to_matrix(hq)
    else:
        assert False

    rot = rot.data.cpu().numpy()

    v = rot.T.dot(v[:, None]).ravel()
    return v


def rotation_from_quaternion(quaternion, separate=False):
    # if 1.0 - quaternion[0] < 1e-8:
    if np.abs(1.0 - quaternion[0]) < 1e-6 or np.abs(1 + quaternion[0]) < 1e-6:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        angle = 0.0
    else:
        angle = 2 * math.acos(quaternion[0])
        axis = quaternion[1:4] / math.sin(angle / 2.0)
        axis /= np.linalg.norm(axis)

    return (axis, angle) if separate else axis * angle


def get_head_vel(head_pose, dt=1 / 30):
    # get head velocity
    head_vels = []
    head_qpos = head_pose[0]

    for i in range(head_pose.shape[0] - 1):
        curr_qpos = head_pose[i, :]
        next_qpos = head_pose[i + 1, :]
        v = (next_qpos[:3] - curr_qpos[:3]) / dt
        v = transform_vec(v.data.cpu().numpy(), curr_qpos[3:7], "heading")  # get velocity within the body frame

        qrel = transforms.quaternion_multiply(next_qpos[3:7], transforms.quaternion_invert(curr_qpos[3:7]))
        axis, angle = rotation_from_quaternion(qrel, True)

        if angle > np.pi:  # -180 < angle < 180
            angle -= 2 * np.pi  #
        elif angle < -np.pi:
            angle += 2 * np.pi

        rv = (axis * angle) / dt
        rv = transform_vec(rv, curr_qpos[3:7], "root")

        head_vels.append(np.concatenate((v, rv)))

    head_vels.append(head_vels[-1].copy())  # copy last one since there will be one less through finite difference
    head_vels = np.vstack(head_vels)
    return head_vels


def determine_floor_height_and_contacts(body_joint_seq, fps):
    """
    Input: body_joint_seq N x 21 x 3 numpy array
    Contacts are N x 4 where N is number of frames and each row is left heel/toe, right heel/toe
    """
    num_frames = body_joint_seq.shape[0]
    assert fps == OUT_FPS

    # compute toe velocities
    root_seq = body_joint_seq[:, SMPL_JOINTS["hips"], :]
    left_toe_seq = body_joint_seq[:, SMPL_JOINTS["leftToeBase"], :]
    right_toe_seq = body_joint_seq[:, SMPL_JOINTS["rightToeBase"], :]
    left_toe_vel = np.linalg.norm(left_toe_seq[1:] - left_toe_seq[:-1], axis=1)
    left_toe_vel = np.append(left_toe_vel, left_toe_vel[-1])
    right_toe_vel = np.linalg.norm(right_toe_seq[1:] - right_toe_seq[:-1], axis=1)
    right_toe_vel = np.append(right_toe_vel, right_toe_vel[-1])

    if VIZ_PLOTS:
        fig = plt.figure()
        steps = np.arange(num_frames)
        plt.plot(steps, left_toe_vel, "-r", label="left vel")
        plt.plot(steps, right_toe_vel, "-b", label="right vel")
        plt.legend()
        plt.show()
        plt.close()

    # now foot heights (z is up)
    left_toe_heights = left_toe_seq[:, 2]
    right_toe_heights = right_toe_seq[:, 2]
    root_heights = root_seq[:, 2]

    if VIZ_PLOTS:
        fig = plt.figure()
        steps = np.arange(num_frames)
        plt.plot(steps, left_toe_heights, "-r", label="left toe height")
        plt.plot(steps, right_toe_heights, "-b", label="right toe height")
        plt.plot(steps, root_heights, "-g", label="root height")
        plt.legend()
        plt.show()
        plt.close()

    # filter out heights when velocity is greater than some threshold (not in contact)
    all_inds = np.arange(left_toe_heights.shape[0])
    left_static_foot_heights = left_toe_heights[left_toe_vel < FLOOR_VEL_THRESH]
    left_static_inds = all_inds[left_toe_vel < FLOOR_VEL_THRESH]
    right_static_foot_heights = right_toe_heights[right_toe_vel < FLOOR_VEL_THRESH]
    right_static_inds = all_inds[right_toe_vel < FLOOR_VEL_THRESH]

    all_static_foot_heights = np.append(left_static_foot_heights, right_static_foot_heights)
    all_static_inds = np.append(left_static_inds, right_static_inds)

    if VIZ_PLOTS:
        fig = plt.figure()
        steps = np.arange(left_static_foot_heights.shape[0])
        plt.plot(steps, left_static_foot_heights, "-r", label="left static height")
        plt.legend()
        plt.show()
        plt.close()

    # fig = plt.figure()
    # plt.hist(all_static_foot_heights)
    # plt.show()
    # plt.close()

    discard_seq = False
    if all_static_foot_heights.shape[0] > 0:
        cluster_heights = []
        cluster_root_heights = []
        cluster_sizes = []
        # cluster foot heights and find one with smallest median
        clustering = DBSCAN(eps=0.005, min_samples=3).fit(all_static_foot_heights.reshape(-1, 1))
        all_labels = np.unique(clustering.labels_)
        # print(all_labels)
        if VIZ_PLOTS:
            plt.figure()
        min_median = min_root_median = float("inf")
        for cur_label in all_labels:
            cur_clust = all_static_foot_heights[clustering.labels_ == cur_label]
            cur_clust_inds = np.unique(
                all_static_inds[clustering.labels_ == cur_label]
            )  # inds in the original sequence that correspond to this cluster
            if VIZ_PLOTS:
                plt.scatter(cur_clust, np.zeros_like(cur_clust), label="foot %d" % (cur_label))
            # get median foot height and use this as height
            cur_median = np.median(cur_clust)
            cluster_heights.append(cur_median)
            cluster_sizes.append(cur_clust.shape[0])

            # get root information
            cur_root_clust = root_heights[cur_clust_inds]
            cur_root_median = np.median(cur_root_clust)
            cluster_root_heights.append(cur_root_median)
            if VIZ_PLOTS:
                plt.scatter(cur_root_clust, np.zeros_like(cur_root_clust), label="root %d" % (cur_label))

            # update min info
            if cur_median < min_median:
                min_median = cur_median
                min_root_median = cur_root_median

        # print(cluster_heights)
        # print(cluster_root_heights)
        # print(cluster_sizes)
        if VIZ_PLOTS:
            plt.show()
            plt.close()

        floor_height = min_median
        offset_floor_height = floor_height - FLOOR_HEIGHT_OFFSET  # toe joint is actually inside foot mesh a bit

        if DISCARD_TERRAIN_SEQUENCES:
            # print(min_median + TERRAIN_HEIGHT_THRESH)
            # print(min_root_median + ROOT_HEIGHT_THRESH)
            for cluster_root_height, cluster_height, cluster_size in zip(
                cluster_root_heights, cluster_heights, cluster_sizes
            ):
                root_above_thresh = cluster_root_height > (min_root_median + ROOT_HEIGHT_THRESH)
                toe_above_thresh = cluster_height > (min_median + TERRAIN_HEIGHT_THRESH)
                cluster_size_above_thresh = cluster_size > int(CLUSTER_SIZE_THRESH * fps)
                if root_above_thresh and toe_above_thresh and cluster_size_above_thresh:
                    discard_seq = True
                    print("DISCARDING sequence based on terrain interaction!")
                    break
    else:
        floor_height = offset_floor_height = 0.0

    # now find contacts (feet are below certain velocity and within certain range of floor)
    # compute heel velocities
    left_heel_seq = body_joint_seq[:, SMPL_JOINTS["leftFoot"], :]
    right_heel_seq = body_joint_seq[:, SMPL_JOINTS["rightFoot"], :]
    left_heel_vel = np.linalg.norm(left_heel_seq[1:] - left_heel_seq[:-1], axis=1)
    left_heel_vel = np.append(left_heel_vel, left_heel_vel[-1])
    right_heel_vel = np.linalg.norm(right_heel_seq[1:] - right_heel_seq[:-1], axis=1)
    right_heel_vel = np.append(right_heel_vel, right_heel_vel[-1])

    left_heel_contact = left_heel_vel < CONTACT_VEL_THRESH
    right_heel_contact = right_heel_vel < CONTACT_VEL_THRESH
    left_toe_contact = left_toe_vel < CONTACT_VEL_THRESH
    right_toe_contact = right_toe_vel < CONTACT_VEL_THRESH

    # compute heel heights
    left_heel_heights = left_heel_seq[:, 2] - floor_height
    right_heel_heights = right_heel_seq[:, 2] - floor_height
    left_toe_heights = left_toe_heights - floor_height
    right_toe_heights = right_toe_heights - floor_height

    left_heel_contact = np.logical_and(left_heel_contact, left_heel_heights < CONTACT_ANKLE_HEIGHT_THRESH)
    right_heel_contact = np.logical_and(right_heel_contact, right_heel_heights < CONTACT_ANKLE_HEIGHT_THRESH)
    left_toe_contact = np.logical_and(left_toe_contact, left_toe_heights < CONTACT_TOE_HEIGHT_THRESH)
    right_toe_contact = np.logical_and(right_toe_contact, right_toe_heights < CONTACT_TOE_HEIGHT_THRESH)

    contacts = np.zeros((num_frames, len(SMPL_JOINTS)))
    contacts[:, SMPL_JOINTS["leftFoot"]] = left_heel_contact
    contacts[:, SMPL_JOINTS["leftToeBase"]] = left_toe_contact
    contacts[:, SMPL_JOINTS["rightFoot"]] = right_heel_contact
    contacts[:, SMPL_JOINTS["rightToeBase"]] = right_toe_contact

    # hand contacts
    left_hand_contact = detect_joint_contact(
        body_joint_seq, "leftHand", floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH
    )
    right_hand_contact = detect_joint_contact(
        body_joint_seq, "rightHand", floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH
    )
    contacts[:, SMPL_JOINTS["leftHand"]] = left_hand_contact
    contacts[:, SMPL_JOINTS["rightHand"]] = right_hand_contact

    # knee contacts
    left_knee_contact = detect_joint_contact(
        body_joint_seq, "leftLeg", floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH
    )
    right_knee_contact = detect_joint_contact(
        body_joint_seq, "rightLeg", floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH
    )
    contacts[:, SMPL_JOINTS["leftLeg"]] = left_knee_contact
    contacts[:, SMPL_JOINTS["rightLeg"]] = right_knee_contact

    return offset_floor_height, contacts, discard_seq


def detect_joint_contact(body_joint_seq, joint_name, floor_height, vel_thresh, height_thresh):
    # calc velocity
    joint_seq = body_joint_seq[:, SMPL_JOINTS[joint_name], :]
    joint_vel = np.linalg.norm(joint_seq[1:] - joint_seq[:-1], axis=1)
    joint_vel = np.append(joint_vel, joint_vel[-1])
    # determine contact by velocity
    joint_contact = joint_vel < vel_thresh
    # compute heights
    joint_heights = joint_seq[:, 2] - floor_height
    # compute contact by vel + height
    joint_contact = np.logical_and(joint_contact, joint_heights < height_thresh)

    return joint_contact


def quat_fk_torch(lrot_mat, lpos, parents):
    # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
    # lpos: N X J X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
    lrot = transforms.matrix_to_quaternion(lrot_mat)
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(transforms.quaternion_apply(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]])
        gr.append(transforms.quaternion_multiply(gr[parents[i]], lrot[..., i : i + 1, :]))

    return torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)


def mat_fk_torch(lrot, lpos, parents):
    # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
    # lpos: N X J X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :, :]]
    for i in range(1, len(parents)):
        gp.append(torch.matmul(gr[parents[i]], lpos[..., i : i + 1, :][..., None])[..., 0] + gp[parents[i]])
        gr.append(torch.matmul(gr[parents[i]], lrot[..., i : i + 1, :, :]))

    return torch.cat(gr, dim=-3), torch.cat(gp, dim=-2)


def quat_ik_torch(grot_mat, parents):
    # grot: T X J X 3 X 3

    grot = transforms.matrix_to_quaternion(grot_mat)  # T X J X 4

    res = torch.cat(
        [
            grot[..., :1, :],
            transforms.quaternion_multiply(transforms.quaternion_invert(grot[..., parents[1:], :]), grot[..., 1:, :]),
        ],
        dim=-2,
    )  # T X J X 4

    res_mat = transforms.quaternion_to_matrix(res)  # T X J X 3 X 3

    return res_mat


def mat_ik_torch(grot, parents):
    # grot: T X J X 3 X 3

    res = torch.cat(
        [
            grot[..., :1, :, :],
            torch.matmul(torch.inverse(grot[..., parents[1:], :, :]), grot[..., 1:, :, :]),
        ],
        dim=-3,
    )  # T X J X 3 x 3
    return res
