import torch
from .rotation_conversions import matrix_to_axis_angle, axis_angle_to_matrix


def matrix_to_pca(hand_pose, hand_components):
    # hand_pose: (B, 15, 3, 3), hand_components: (C, 45)
    hand_pose = matrix_to_axis_angle(hand_pose)  # (B, 15, 3)
    hand_pose = hand_pose.flatten(-2, -1)  # (B, 45)
    hand_pose = torch.einsum("...i,ij->...j", [hand_pose, hand_components.T])
    diags = hand_components.square().sum(-1)  # divide by eigenvalues
    hand_pose = hand_pose / diags
    return hand_pose


def pca_to_matrix(hand_pose, hand_components):
    # hand_pose: (B, C), hand_components: (C, 45)
    hand_pose = torch.einsum("...i,ij->...j", [hand_pose, hand_components])  # (B, 45)
    hand_pose = hand_pose.unflatten(-1, (15, 3))  # (B, 15, 3)
    hand_pose = axis_angle_to_matrix(hand_pose)  # (B, 15, 3, 3)
    return hand_pose
