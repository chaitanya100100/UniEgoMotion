import json
import os

import cv2
import joblib
import numpy as np
import pandas as pd
import projectaria_tools.core.mps as mps
import torch
from loguru import logger
from PIL import Image
from projectaria_tools.core import calibration, data_provider
from projectaria_tools.core.mps.utils import get_nearest_pose
from scipy.spatial.transform import Rotation


BODY_JOINT_NAMES = [
    "nose",
    "left-eye",
    "right-eye",
    "left-ear",
    "right-ear",
    "left-shoulder",
    "right-shoulder",
    "left-elbow",
    "right-elbow",
    "left-wrist",
    "right-wrist",
    "left-hip",
    "right-hip",
    "left-knee",
    "right-knee",
    "left-ankle",
    "right-ankle",
]

HAND_JOINTS_NAME = [
    "right_wrist",
    "right_thumb_1",
    "right_thumb_2",
    "right_thumb_3",
    "right_thumb_4",
    "right_index_1",
    "right_index_2",
    "right_index_3",
    "right_index_4",
    "right_middle_1",
    "right_middle_2",
    "right_middle_3",
    "right_middle_4",
    "right_ring_1",
    "right_ring_2",
    "right_ring_3",
    "right_ring_4",
    "right_pinky_1",
    "right_pinky_2",
    "right_pinky_3",
    "right_pinky_4",
    "left_wrist",
    "left_thumb_1",
    "left_thumb_2",
    "left_thumb_3",
    "left_thumb_4",
    "left_index_1",
    "left_index_2",
    "left_index_3",
    "left_index_4",
    "left_middle_1",
    "left_middle_2",
    "left_middle_3",
    "left_middle_4",
    "left_ring_1",
    "left_ring_2",
    "left_ring_3",
    "left_ring_4",
    "left_pinky_1",
    "left_pinky_2",
    "left_pinky_3",
    "left_pinky_4",
]


ARIA_INTRINSICS = [[150.0, 0.0, 255.5], [0.0, 150.0, 255.5], [0.0, 0.0, 1.0]]


def parse_kp3d(kp3d, joint_names):
    """Parse json kp3d data into poses and vis lists.

    Args:
        kp3d (dict): kp3d json data
        joint_names (list): list of joint names
    Returns:
        np.array: 3D poses Jx4 where last column is number of views used for 3D
    """
    poses = []
    for keyp in joint_names:
        if keyp in kp3d:
            assert kp3d[keyp]["num_views_for_3d"] > 0, f"num_views_for_3d is 0 for {keyp}"
            poses.append([kp3d[keyp]["x"], kp3d[keyp]["y"], kp3d[keyp]["z"], kp3d[keyp]["num_views_for_3d"]])  # visible
        else:
            poses.append([0, 0, 0, 0])  # not visible
    return np.array(poses)


def parse_kp2d(kp2d, joint_names):
    """Parse json kp2d data into poses and vis lists.

    Args:
        kp2d (dict): kp2d json data
        joint_names (list): list of joint names
    Returns:
        np.array: 2D poses Jx3 where last column is type flag
        Type flag: 0 - not visible, 1 - auto, 2 - manual
    """
    annot_type = {"auto": 1, "manual": 2}
    poses = []
    for keyp in joint_names:
        if keyp in kp2d:
            poses.append([kp2d[keyp]["x"], kp2d[keyp]["y"], annot_type[kp2d[keyp]["placement"]]])
        else:
            poses.append([0, 0, 0])  # not visible
    return np.array(poses)


def get_ego_aria_cam_name(take):
    """Get the name of the ego camera from the take metadata.

    Args:
        take (dict): take metadata
    Returns:
        str: name of the ego camera (e.g. "aria02")
    """
    ego_cam_names = [cam for cam in take["frame_aligned_videos"].keys() if "aria" in cam.lower()]
    assert len(ego_cam_names) == 1, f"Found {len(ego_cam_names)} Aria videos in {take['take_name']}"
    return ego_cam_names[0]


def get_cam_names(take):
    """Get the name of the exo cameras from the take metadata.

    Args:
        take (dict): take metadata
    Returns:
        list: names of the exo cameras (e.g. ["cam01", "cam02"], ["gp01", "gp02"])
    """
    # cam_names = [cam for cam in take["frame_aligned_videos"].keys() if cam.lower().startswith("cam")]
    cam_names = [
        cam
        for cam in take["frame_aligned_videos"].keys()
        if cam.lower().startswith("cam") or cam.lower().startswith("gp")
    ]
    assert len(cam_names) > 0, f"Found {len(cam_names)} Exo videos in {take['take_name']}"
    return cam_names


def get_extrinsics_and_raw_intrinsics(calib_file):
    """Get extrinsics and intrinsics for raw distorted videos.
    Inspired from https://github.com/facebookresearch/Ego4d/issues/331#issuecomment-2174224124

    Args:
        calib_file (str): path to the calibration csv file
    Returns:
        dict: extrinsics {cam: 3x4}
        dict: distortion coefficients {cam: 4}
        dict: intrinsics {cam: 3x3}
    """

    gopro_calibs_df = pd.read_csv(calib_file)
    extrinsics, intrinsics, distortion_coeffs = {}, {}, {}

    for cam in gopro_calibs_df["cam_uid"].values:
        row = gopro_calibs_df[gopro_calibs_df["cam_uid"] == cam].iloc[0]

        intrinsics[cam] = np.array(
            [
                [row["intrinsics_0"], 0, row["intrinsics_2"]],
                [0, row["intrinsics_1"], row["intrinsics_3"]],
                [0, 0, 1],
            ]
        )
        distortion_coeffs[cam] = np.array(
            [
                row["intrinsics_4"],
                row["intrinsics_5"],
                row["intrinsics_6"],
                row["intrinsics_7"],
            ]
        )
        trans = np.array([row["tx_world_cam"], row["ty_world_cam"], row["tz_world_cam"]])
        quat = np.array(
            [
                row["qx_world_cam"],
                row["qy_world_cam"],
                row["qz_world_cam"],
                row["qw_world_cam"],
            ]
        )
        rotmat = Rotation.from_quat(quat).as_matrix()
        rotmat = np.linalg.inv(rotmat)
        trans = -rotmat @ trans[:, None]
        extrinsics[cam] = np.concatenate([rotmat, trans], axis=1)
        # extrinsics[cam] = np.concatenate([rotmat, trans[:, None]], axis=1)

    return extrinsics, intrinsics, distortion_coeffs


def get_extrinsics_and_undistort_intrinsics(cam_pos_path):
    """Get extrinsics and undistorted intrinsics for the cameras.
    Here, we use pose data provided by ego_pose challenge which is in undistorted space.
    """
    with open(cam_pos_path) as f:
        cam_pos = json.load(f)
    extrinsics, intrinsics, distortion_coeffs = {}, {}, {}
    for cam in cam_pos.keys():
        if cam == "metadata":
            continue
        intrinsics[cam] = np.array(cam_pos[cam]["camera_intrinsics"])
        if "aria" in cam.lower():
            x = cam_pos[cam]["camera_extrinsics"]
            extrinsics[cam] = np.array([x[str(i)] for i in range(len(x))])
        else:
            extrinsics[cam] = np.array(cam_pos[cam]["camera_extrinsics"])
            distortion_coeffs[cam] = np.array(cam_pos[cam]["distortion_coeffs"])
    return extrinsics, intrinsics, distortion_coeffs


def smooth_egopose_annotation(ann):
    """Smooths the pose annotation by applying a 1D convolutional kernel.

    Args:
        ann (dict): {frame: np.array} pose annotation of 2d or 3d keypoints.
    """
    frames = np.array(sorted(ann.keys()))
    vals = np.stack([ann[f] for f in frames])  # TJ3 for kp2d
    assert len(vals.shape) == 3, f"Expected 3D array, got {vals.shape}"
    weight = np.zeros(vals.shape[:2])  # TJ
    # kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    kernel = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1])
    l = kernel.shape[0]
    t = frames.shape[0]

    # smooth_vals = np.concatenate([vals[0:1], vals, vals[-1:]])
    # smooth_vals = kernel[0] * smooth_vals[:-2] + kernel[1] * smooth_vals[1:-1] + kernel[2] * smooth_vals[2:]
    temp_vals = np.concatenate([vals[0 : l // 2], vals, vals[-(l // 2) :]], axis=0)
    smooth_vals = np.zeros_like(vals)
    for i in range(kernel.shape[0]):
        w = (temp_vals[i : i + t, :, -1]).astype(bool).astype(np.float64)  # TJ
        w = w * kernel[i]
        weight += w
        smooth_vals += w[:, :, None] * temp_vals[i : i + t]  # TJ3

    smooth_vals = smooth_vals / (weight[:, :, None] + 1.0e-7)

    smooth_vals[..., -1] = vals[..., -1]  # Don't smooth the visibility or num cameras
    ann = {f: smooth_vals[i] for i, f in enumerate(frames)}
    return ann


def smooth_sequence(vals):
    """Smooth a sequence of values using a Gaussian kernel."""
    kernel = torch.tensor([0.05, 0.25, 0.4, 0.25, 0.05])
    # kernel = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1])
    l = kernel.shape[0]
    t = vals.shape[0]

    # smooth_vals = np.concatenate([vals[0:1], vals, vals[-1:]])
    # smooth_vals = kernel[0] * smooth_vals[:-2] + kernel[1] * smooth_vals[1:-1] + kernel[2] * smooth_vals[2:]
    temp_vals = torch.concatenate([vals[0 : l // 2], vals, vals[-(l // 2) :]], dim=0)
    smooth_vals = torch.zeros_like(vals)
    for i in range(kernel.shape[0]):
        smooth_vals += kernel[i] * temp_vals[i : i + t]  # TJ3
    return smooth_vals


def ego_extri_to_egoego_head_traj(ego_extri):
    head_traj = {}
    for fi, extri_fi in ego_extri.items():

        tsfm = np.eye(4)
        tsfm[:3, :4] = extri_fi
        tsfm = np.linalg.inv(tsfm)

        # Make it consistent with egoego
        bla = np.array(
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        )
        tsfm[:3, :3] = tsfm[:3, :3] @ bla

        # # Approximate height of the device from the ground.
        # # Please replace this with the actual height of the device from the ground.
        # tsfm[2, 3] += 1.6

        head_traj[fi] = tsfm
    return head_traj


def multiInterp2_egopose(x, xp, fp):
    """Interpolate 2d or 3d tensors."""
    # x: (N,)
    # xp: (M,...)
    # fp: (M,...)
    j = np.searchsorted(xp, x) - 1
    d = (x - xp[j]) / (xp[j + 1] - xp[j])
    if len(fp.shape) == 2:
        d = d[..., None]
    elif len(fp.shape) == 3:
        d = d[..., None, None]
    else:
        raise ValueError(f"Expected 2D or 3D array, got {len(fp.shape)}D.")
    res = (1 - d) * fp[j] + fp[j + 1] * d
    # don't interpolate visibility or num cameras
    res[..., -1] = np.minimum(fp[j][..., -1], fp[j + 1][..., -1])
    return res


def egopose_infill_missing(ann):
    """Infill annotations from 10fps to 30fps.

    Args:
        ann (dict): {frame: np.array} pose annotation of 2d or 3d keypoints.
    """
    frames = sorted([f for f in ann.keys()])
    missing_frames = list(set(range(frames[0], frames[-1] + 1)) - set(frames))
    if len(missing_frames) == 0:
        return ann

    # Remove missing frames that are too far from the existing frames
    frames = np.array(frames)
    missing_frames = np.array(missing_frames)
    indices = np.searchsorted(frames, missing_frames)
    missing_left = frames[indices - 1]
    missing_right = frames[indices]
    dist = np.maximum(missing_frames - missing_left, missing_right - missing_frames)
    missing_frames = missing_frames[dist < 15]  # 15 frames = 0.5s
    missing_frames = missing_frames.tolist()

    # print(f"Missing frames for egopose infilling: {missing_frames}")
    avail_nf = len(frames)
    filled_nf = len(missing_frames)
    unfilled_nf = frames[-1] - frames[0] + 1 - avail_nf - filled_nf
    print(
        f"Egopose infilling. Available frames: {avail_nf}, Filled frames: {filled_nf}, Unfilled frames: {unfilled_nf}."
    )

    data_dim = ann[frames[0]].shape[-1] - 1
    assert data_dim in [2, 3], f"Expected 3D or 2D data, got {data_dim}D."
    missing_data = multiInterp2_egopose(np.array(missing_frames), np.array(frames), np.stack([ann[f] for f in frames]))
    for f, d in zip(missing_frames, missing_data):
        ann[f] = d
    return ann


def undistort_exocam(image, intrinsics, distortion_coeffs, dimension=(3840, 2160)):
    """
    Undistort an image from the exocentric camera.
    See https://github.com/facebookresearch/Ego4d/issues/331#issuecomment-2174224124
    """
    DIM = dimension
    dim2 = None
    dim3 = None
    balance = 0.8
    # Load the distortion parameters
    distortion_coeffs = distortion_coeffs
    # Load the camera intrinsic parameters
    intrinsics = intrinsics

    dim1 = image.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort

    # Change the calibration dim dynamically (bouldering cam01 and cam04 are verticall for examples)
    if DIM[0] != dim1[0]:
        DIM = (DIM[1], DIM[0])

    assert (
        dim1[0] / dim1[1] == DIM[0] / DIM[1]
    ), "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = intrinsics * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image.
    # OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        scaled_K, distortion_coeffs, dim2, np.eye(3), balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, distortion_coeffs, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistorted_image, new_K


def get_aria_intrinsics_extrinsics_and_calib(data_dir, take_name, ego_cam):
    """Load Aria intrinsics, extrinsics, and calibration for a take.
    Calibration object is later used to undistort the raw aria video frames.
    """
    rgb_stream_label, calib_fix = "camera-rgb", True
    # stored_extri_path = f"{take_dir}/trajectory/aria_extrinsics.json"
    stored_extri_path = f"{data_dir}/ee4d_motion/{take_name}/aria_extrinsics.json"
    take_dir = f"{data_dir}/takes/{take_name}"

    # Get VRS file and calibration object
    vrs_file = f"{take_dir}/{ego_cam}_noimagestreams.vrs"
    assert os.path.exists(vrs_file), f"VRS file not found: {vrs_file}"
    vrs_data_provider = data_provider.create_vrs_data_provider(vrs_file)
    # rgb_stream_id = StreamId("214-1")
    # rgb_stream_label = vrs_data_provider.get_label_from_stream_id(rgb_stream_id)
    device_calibration = vrs_data_provider.get_device_calibration()
    src_calib = device_calibration.get_camera_calib(rgb_stream_label)

    # Fix calibration for Aria camera
    # https://github.com/EGO4D/ego-exo4d-egopose/blob/2494c0d192b92784525342df6c6828e3aad04c2e/handpose/data_preparation/main.py#L295
    if calib_fix:
        proj_params = src_calib.projection_params()
        proj_params[0] /= 2
        proj_params[1] = (proj_params[1] - 0.5 - 32) / 2
        proj_params[2] = (proj_params[2] - 0.5 - 32) / 2

        src_calib = calibration.CameraCalibration(
            src_calib.get_label(),
            src_calib.model_name(),
            proj_params,
            src_calib.get_transform_device_camera(),
            src_calib.get_image_size()[0],
            src_calib.get_image_size()[1],
            src_calib.get_valid_radius(),
            src_calib.get_max_solid_angle(),
            src_calib.get_serial_number(),
        )

    # Fixed Aria intrinsics for undistorted image
    intri = np.array(ARIA_INTRINSICS)

    # Per frame extrinsics
    if os.path.exists(stored_extri_path):
        # If stored extrinsics exist, load them
        with open(stored_extri_path) as f:
            extri = json.load(f)
        extri = {int(k): np.array(v) for k, v in extri.items()}
    else:
        # If stored extrinsics do not exist, calculate them and store them
        T_device_rgb_camera = src_calib.get_transform_device_camera()
        closed_loop_path = f"{take_dir}/trajectory/closed_loop_trajectory.csv"
        closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)
        start_ns = int(closed_loop_traj[0].tracking_timestamp.total_seconds() * 1e9)
        end_ns = int(closed_loop_traj[-1].tracking_timestamp.total_seconds() * 1e9)

        extri = {}
        for fidx, cur_ns in enumerate(np.arange(start_ns + 1, end_ns + 1, 1e9 / 30.005)):
            pose_info = get_nearest_pose(closed_loop_traj, cur_ns)
            T_world_device = pose_info.transform_world_device
            T_world_rgb_camera = T_world_device @ T_device_rgb_camera
            extri[fidx] = np.linalg.inv(T_world_rgb_camera.to_matrix())[:3, :4]

        extri_to_store = {k: np.round(v, 6).tolist() for k, v in extri.items()}
        with open(stored_extri_path, "w") as f:
            json.dump(extri_to_store, f)
        logger.info(f"Stored Aria extrinsics in {stored_extri_path}")

    return intri, extri, src_calib


def undistort_aria(image_array, aria_calib):
    sensor_name, focal_length, size, calib_fix = "camera-rgb", 150, 512, True

    assert isinstance(image_array, np.ndarray), f"Expected np.ndarray, got {type(image_array)}"

    # create output calibration: a linear model of image size 512x512 and focal length 150
    # Invisible pixels are shown as black.
    dst_calib = calibration.get_linear_camera_calibration(size, size, focal_length, sensor_name)

    # distort image
    rectified_array = calibration.distort_by_calibration(image_array, dst_calib, aria_calib)

    return rectified_array
    return (rectified_array, dst_calib.get_principal_point(), dst_calib.get_focal_lengths())


def world_to_cam(kpts, extri):
    """Transform points from world coordinate system to camera coordinate system.

    Args:
        kpts: (N,3) array of 3D points in world coordinate system
        extri: (3,4) array of extrinsic matrix
    Returns:
        (N,3) array of 3D points in camera coordinate system
    """
    new_kpts = kpts.copy()
    new_kpts = np.append(new_kpts, np.ones((new_kpts.shape[0], 1)), axis=1).T  # (4,N)
    new_kpts = (extri @ new_kpts).T  # (N,3)
    return new_kpts


def cam_to_img(kpts, intri):
    """Transform points from camera coordinate system to image coordinate system.

    Args:
        kpts: (N,3) array of 3D points in camera coordinate system
        intri: (3,3) array of intrinsic matrix
    Returns:
        (N,2) array of 2D points in image coordinate system
    """

    new_kpts = kpts.copy()
    new_kpts = intri @ new_kpts.T  # (3,N)
    new_kpts = new_kpts / new_kpts[2, :]
    new_kpts = new_kpts[:2, :].T
    return new_kpts


def mat_3x4_to_4x4(mat):
    assert mat.shape[0] == 3 and mat.shape[1] == 4
    if isinstance(mat, torch.Tensor):
        return torch.cat([mat, torch.tensor([[0, 0, 0, 1]], device=mat.device)], dim=0)
    return np.concatenate([mat, np.array([[0, 0, 0, 1]])], axis=0)


def downscale_image(image):
    """Downscale camera image such that shortest side is 448 pixels."""
    if image.shape[0] == image.shape[1]:  # ego image
        ow, oh = (1408, 1408)
        rw, rh = (448, 448)
    else:  # exo image
        ow, oh = (3840, 2160)
        rw, rh = (796, 448)
        if image.shape[0] != oh:
            ow, oh = oh, ow
            rw, rh = rh, rw

    assert tuple(image.shape[:2]) == (oh, ow), f"Image shape {image.shape} does not match expected shape ({oh}, {ow})."
    return cv2.resize(image, (rw, rh))


def upscale_image(image):
    """Upscale downscaled camera image to original size."""
    if image.shape[0] == image.shape[1]:  # ego image
        ow, oh = (1408, 1408)
        rw, rh = (448, 448)
    else:  # exo image
        ow, oh = (3840, 2160)
        rw, rh = (796, 448)
        if image.shape[0] != rh:
            ow, oh = oh, ow
            rw, rh = rh, rw

    assert tuple(image.shape[:2]) == (rh, rw), f"Image shape {image.shape} does not match expected shape ({rh}, {rw})."
    return cv2.resize(image, (ow, oh), interpolation=cv2.INTER_LINEAR)


def get_video_meta(video_path):
    cap = cv2.VideoCapture(video_path)
    if not (cap.isOpened()):
        return False
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    vdur = nframes / fps
    cap.release()
    return {
        "height": int(height),
        "width": int(width),
        "num_frames": nframes,
        "fps": fps,
        "duration": vdur,
    }


def is_take_valid(take):
    if not take["validated"]:
        return False
    if not take["has_trimmed_trajectory"]:
        return False
    if not take["has_trimmed_vrs"]:
        return False
    return True


def egopose_take_uid_to_split(data_dir):
    egopose_uid2split = {}
    for split in ["train", "val"]:
        _pos_dir = f"{data_dir}/annotations/ego_pose/{split}/body/annotation"
        _uids = os.listdir(_pos_dir)
        _uids = [os.path.splitext(x)[0] for x in _uids]
        egopose_uid2split.update({uid: split for uid in _uids})
    egopose_uid2split.pop("b1b794e8-7839-46ab-b05f-f4b1c16d5420")  # metadata not available for this take
    return egopose_uid2split


def get_pointcloud(pointcloud_path, inverse_distance_std_threshold=0.001, distance_std_threshold=0.15):
    """
    Inspired from https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/core_code_snippets/mps
    Get point cloud in world coordinate system.
    Args: 
        pointcloud_path (str): path to the point cloud file
        inverse_distance_std_threshold (float): 
        distance_std_threshold (float): 
    Return:
        pointcloud (numpy array): point clout of a take in world coordinate system. \
            Shape is (N, 3). N is the number of points and 3 is x, y and z.
    """

    # pointcloud_path = f"{root}/{take_meta['root_dir']}/trajectory/semidense_points.csv.gz"
    # if not os.path.exists(pointcloud_path):  # check capture point cloud
    #     pointcloud_path = f"{root}/{take_meta['capture']['root_dir']}/trajectory/semidense_points.csv.gz"
    # assert os.path.exists(pointcloud_path), f"Point cloud not found: {pointcloud_path}"

    df_pointcloud = pd.read_csv(pointcloud_path)
    # filter the point cloud using thresholds on the inverse depth and distance standard deviation
    df_pointcloud = df_pointcloud[
        (df_pointcloud["inv_dist_std"] <= inverse_distance_std_threshold)
        & (df_pointcloud["dist_std"] <= distance_std_threshold)
    ]

    pointcloud = df_pointcloud[["px_world", "py_world", "pz_world"]].values
    return pointcloud
