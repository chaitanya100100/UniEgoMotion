import json
import os
import shutil
import tempfile

import joblib
import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

from dataset.egoexo4d_utils import (
    BODY_JOINT_NAMES,
    HAND_JOINTS_NAME,
    cam_to_img,
    egopose_infill_missing,
    egopose_take_uid_to_split,
    get_aria_intrinsics_extrinsics_and_calib,
    get_ego_aria_cam_name,
    get_extrinsics_and_raw_intrinsics,
    get_video_meta,
    is_take_valid,
    parse_kp2d,
    parse_kp3d,
    smooth_egopose_annotation,
    undistort_aria,
    undistort_exocam,
    upscale_image,
    world_to_cam,
)
from dataset.video_readers import SequentialVideoReader


class EgoExo4D_Take_Dataset(Dataset):
    def __init__(self, data_dir, lowres=False, fps=10):
        """Initialize the dataset.
        Args:
            data_dir (str): Path to egoexo4d dataset directory.
            lowres (bool): If True, load low-resolution images and then upsample.
            fps (int): Frames per second for the video. EgoExo4D is at 30fps. EE4D-Motion dataset is at 10fps.
        """
        super().__init__()
        self.data_dir = data_dir
        self.undistort = True
        self.load_ego = True
        self.lowres = lowres
        self.fps = fps
        assert self.fps in [10, 30], "Only 10 or 30 fps is supported. You may change this with caution."
        assert self.load_ego, "Set load_ego to True for consistent extrinsics."
        if self.lowres:
            logger.warning("Low-resolution mode is on.")

        # Load generic metadata for all takes
        with open(f"{self.data_dir}/takes.json") as f:
            metadata = json.load(f)
        self.metadata = {x["take_name"]: x for x in metadata if is_take_valid(x)}
        self.uid2name = {x["take_uid"]: x["take_name"] for x in metadata}

        # These two splits are different.
        # self.egopose_uid2split = egopose_take_uid_to_split(self.data_dir)
        # with open(f"{data_dir}/annotations/splits.json", "r") as f:
        #     self.orig_split = json.load(f)

        # The following gets populated when you call load_take method.
        # Most of the other methods rely on this data.
        # Make sure to understand load_take method first before anything else.
        self.take_data = None
        self.video_readers = {}
        self.local_dir = None

    def __len__(self):
        return len(self.take_data["frames"])

    @property
    def all_cams(self):
        if self.load_ego:
            return self.take_data["exo_cams"] + [self.take_data["ego_cam"]]
        else:
            return self.take_data["exo_cams"]

    def __getitem__(self, index):
        # Avoid using this method directly. If index is not provided sequentially, it may not work as expected.
        # Use get_rgb_generator instead. See dataset visualization script for example usage.
        frame_idx = self.take_data["frames"][index]
        if len(self.video_readers) == 0:
            self.set_video_readers()
        rgb = self.get_rgb_frames(frame_idx)
        imgs_data = self.undistort_loaded_frames(rgb)
        imgs_data["frame_idx"] = frame_idx
        return imgs_data

    def copy_to_local(self, all_cams=None):
        all_cams = self.all_cams if all_cams is None else all_cams
        self.local_dir = tempfile.mkdtemp()
        rgb_paths = self.get_rgb_video_paths()
        for cam in all_cams:
            rgb_path = rgb_paths[cam]
            local_path = os.path.join(self.local_dir, os.path.basename(rgb_path))
            shutil.copy2(rgb_path, local_path)
            os.sync()
        logger.info(f"Copied videos locally to {self.local_dir}.")

    def delete_local_dir(self):
        if self.local_dir is not None:
            shutil.rmtree(self.local_dir)
            logger.info(f"Deleted local dir {self.local_dir}.")
            self.local_dir = None

    def get_rgb_generator(self, all_cams=None, num_workers=0, local_copy=False):
        """Primary generator to iterate on framewise data sequentially.

        Args:
            all_cams (list, optional): List of all cameras to include. None means all cameras will be used.
            num_workers (int, optional): Number of worker processes for data loading. Defaults to 0.
            local_copy (bool, optional): Whether to copy videos locally for faster loading. Defaults to False.

        Since dataloading is sequential, num_workers more than 2-3 actually slows down the loading. I recommend keeping it 0.
        """

        self.close_video_readers()

        if local_copy:
            self.copy_to_local(all_cams)

        # Basic way of generating dataloader doesn't work because the video readers are not deep copied.
        # worker_init_fn is a workaround to initialize different video readers for each worker.
        def worker_init_fn(worker_id):
            worker_info = torch.utils.data.get_worker_info()
            dataset = worker_info.dataset
            dataset.set_video_readers(all_cams=all_cams)

        if num_workers == 0:
            self.set_video_readers(all_cams=all_cams)
            worker_init_fn = None

        dataloader = torch.utils.data.DataLoader(
            self, batch_size=1, num_workers=num_workers, collate_fn=lambda x: x[0], worker_init_fn=worker_init_fn
        )
        for imgs_data in dataloader:
            yield imgs_data
        self.close_video_readers()

    def set_video_readers(self, all_cams=None):
        assert len(self.video_readers.keys()) == 0, "Video readers already opened."
        take_name = self.take_data["take_name"]
        take_meta = self.metadata[take_name]
        all_cams = self.all_cams if all_cams is None else all_cams

        frames_dir = {cam: f"{self.data_dir}/ee4d_motion/{take_name}/{cam}" for cam in all_cams}
        for cam in all_cams:
            if os.path.exists(frames_dir[cam]):
                logger.warning(f"Frames already extracted for {frames_dir[cam]}.")
                continue
            is_ego = cam == self.take_data["ego_cam"]
            cam_key = "rgb" if is_ego else "0"
            rel_path = take_meta["frame_aligned_videos"][cam][cam_key]["relative_path"]
            vid_path = f"{self.data_dir}/{take_meta['root_dir']}/{rel_path}"
            if self.lowres:
                vid_path = vid_path.replace("frame_aligned_videos", "frame_aligned_videos/downscaled/448")
            if self.local_dir is not None:
                vid_path = os.path.join(self.local_dir, os.path.basename(vid_path))
                assert os.path.exists(vid_path), f"Video not found at {vid_path}."
            self.video_readers[cam] = SequentialVideoReader(vid_path)
        logger.info(f"Opened {len(self.video_readers)} video readers.")

    def close_video_readers(self):
        if len(self.video_readers.keys()) > 0:
            logger.info(f"Closed {len(self.video_readers)} video readers.")
        for reader in self.video_readers.values():
            reader.close()
        self.video_readers = {}
        self.delete_local_dir()

    def __del__(self):
        # super().__del__()
        self.close_video_readers()

    def get_rgb_frames(self, frame_idx):
        take_name = self.take_data["take_name"]
        all_cams = list(self.video_readers.keys())
        frames_dir = {cam: f"{self.data_dir}/ee4d_motion/{take_name}/{cam}" for cam in all_cams}

        rgb = {}
        for cam in all_cams:
            if os.path.exists(frames_dir[cam]):
                frame_path = f"{frames_dir[cam]}/{frame_idx:06d}.jpg"
                rgb[cam] = np.array(Image.open(frame_path)).astype(np.float32) / 255
                continue

            is_ego = cam == self.take_data["ego_cam"]

            rgb[cam] = self.video_readers[cam].get_frame(frame_idx)
            rgb[cam] = np.array(rgb[cam]).astype(np.float32) / 255
            if self.lowres:
                rgb[cam] = upscale_image(rgb[cam])
            if is_ego:  # ego images are rotated
                rgb[cam] = np.rot90(rgb[cam], 1, (0, 1)).copy()
        return rgb

    def undistort_loaded_frames(self, rgb):
        new_intri = {}
        rgb_mask = {}
        take_name = self.take_data["take_name"]
        for cam in rgb.keys():
            is_ego = cam == self.take_data["ego_cam"]
            if self.undistort:
                if not is_ego:
                    # add alpha channel
                    rgb[cam] = np.concatenate([rgb[cam], np.ones_like(rgb[cam][:, :, :1])], axis=2)
                    rgb[cam], new_intri[cam] = undistort_exocam(
                        rgb[cam], self.take_data["intri_distort"][cam], self.take_data["distort_coeff"][cam]
                    )
                    rgb[cam], rgb_mask[cam] = rgb[cam][:, :, :3], rgb[cam][:, :, 3]
                    if "intri" in self.take_data:
                        a, b = new_intri[cam], self.take_data["intri"][cam]
                        assert np.allclose(a, b), f"intri mismatch {take_name} {cam} {a} {b}"
                else:
                    x = (rgb[cam] * 255).astype(np.uint8)
                    rgb[cam] = undistort_aria(x, self.take_data["ego_calib"])
                    rgb_mask[cam] = undistort_aria(np.ones_like(x[..., 0]) * 255, self.take_data["ego_calib"])
                    rgb[cam] = rgb[cam].astype(np.float32) / 255
                    rgb_mask[cam] = rgb_mask[cam].astype(np.float32) / 255
                    new_intri[cam] = self.take_data["ego_intri"]

        return {
            "rgb": rgb,
            "rgb_mask": rgb_mask,
            "intri": new_intri,
        }

    def get_ego_extri_directly(self, take_name):
        # Get ego extrinsics directly without loading other take data.

        ego_cam = get_ego_aria_cam_name(self.metadata[take_name])
        # root_dir = self.metadata[take_name]["root_dir"]
        # take_path = f"{self.data_dir}/{root_dir}"
        # stored_extri_path = f"{take_path}/trajectory/aria_extrinsics.json"
        stored_extri_path = f"{self.data_dir}/ee4d_motion/{take_name}/aria_extrinsics.json"
        if os.path.exists(stored_extri_path):  # If stored extrinsics exist, load them
            with open(stored_extri_path) as f:
                ego_extri = json.load(f)
            ego_extri = {int(k): np.array(v) for k, v in ego_extri.items()}
        else:
            ego_intri, ego_extri, ego_calib = get_aria_intrinsics_extrinsics_and_calib(
                self.data_dir, take_name, ego_cam
            )
        return ego_extri

    def load_take(self, take_name):
        """Load everything related to a take except video frames.
        Use get_rgb_generator to iterate on RGB frames sequentially.
        """

        take_meta = self.metadata[take_name]
        take_uid = take_meta["take_uid"]

        # Get exo camera intrinsics and extrinsics
        calib_path = f"{self.data_dir}/{take_meta['root_dir']}/trajectory/gopro_calibs.csv"
        extri, intri_distort, distort_coeff = get_extrinsics_and_raw_intrinsics(calib_path)
        # cam_pos_path = self.data_dir / f"annotations/ego_pose/{self.split}/camera_pose/{take_uid}.json"
        # extri2, intri2, distort_coeff2 = get_extrinsics_and_undistort_intrinsics(cam_pos_path)

        # Get camera names. Instead of looking for .mp4 files, we use calibration file for exo cameras.
        # exo_cams = get_cam_names(take_meta)
        exo_cams = list(extri.keys())
        ego_cam = get_ego_aria_cam_name(take_meta)

        # Store take data
        self.take_data = {
            "take_uid": take_uid,
            "task_name": take_meta["task_name"],
            "take_name": take_name,
            "parent_task_name": take_meta["parent_task_name"],
            "exo_cams": exo_cams,
            "ego_cam": ego_cam,
            "best_exo_cam": take_meta["best_exo"],
            "intri_distort": intri_distort,
            "extri": extri,
            "distort_coeff": distort_coeff,
        }

        if self.load_ego:
            ego_intri, ego_extri, ego_calib = get_aria_intrinsics_extrinsics_and_calib(
                self.data_dir, take_name, ego_cam
            )
            self.take_data["ego_intri"] = ego_intri
            self.take_data["ego_extri"] = ego_extri
            self.take_data["ego_calib"] = ego_calib
            self.take_data["extri"][ego_cam] = ego_extri  # to be consistent with exo cameras
            ego_extri_frames = list(ego_extri.keys())

        # Get video information
        self.take_data["video_meta"] = {cam: get_video_meta(v) for cam, v in self.get_rgb_video_paths().items()}
        self.take_data["video_meta"][ego_cam]["height"] = 512  # undistorted aria images are 512x512
        self.take_data["video_meta"][ego_cam]["width"] = 512
        self.take_data["frames"] = list(range(self.take_data["video_meta"][ego_cam]["num_frames"]))
        if self.load_ego:
            if len(ego_extri_frames) != len(self.take_data["frames"]):
                logger.warning(
                    f"Ego extri num frames {len(ego_extri_frames)} and video meta num frames {len(self.take_data['frames'])} are different."
                )
            self.take_data["frames"] = ego_extri_frames
        if self.fps == 10:
            self.take_data["frames"] = self.take_data["frames"][::3]

        logger.info(f"Loaded {take_name} {take_uid} with {len(self.take_data['frames'])} frames at {self.fps} fps.")

        # Get exo undistorted intri.
        self.take_data["intri"] = self.get_exo_cam_intri_undistort()

        # # Load pose annotations
        # self.take_data.update(self.load_egopose_annotations())

        return self.take_data

    # ----------------------------------------------------------------------------
    # EXTRA HELPFUL METHODS
    # ----------------------------------------------------------------------------
    def get_rgb_video_paths(self):
        take_name = self.take_data["take_name"]
        take_meta = self.metadata[take_name]
        all_cams = self.take_data["exo_cams"] + [self.take_data["ego_cam"]]
        rgb_paths = {}
        for cam in all_cams:
            is_ego = cam == self.take_data["ego_cam"]
            cam_key = "rgb" if is_ego else "0"
            rel_path = take_meta["frame_aligned_videos"][cam][cam_key]["relative_path"]
            vid_path = f"{self.data_dir}/{take_meta['root_dir']}/{rel_path}"
            rgb_paths[cam] = vid_path
        return rgb_paths

    def get_exo_cam_intri_undistort(self):
        take_name = self.take_data["take_name"]
        os.makedirs(f"{self.data_dir}/ee4d_motion/{take_name}", exist_ok=True)
        stored_path = f"{self.data_dir}/ee4d_motion/{take_name}/exo_intri_undistort.json"
        if os.path.exists(stored_path):
            with open(stored_path) as f:
                intri = json.load(f)
            intri = {cam: np.array(v) for cam, v in intri.items()}
        else:
            intri = self.__getitem__(0)["intri"]
            self.close_video_readers()
            intri_to_store = {k: np.round(v, 6).tolist() for k, v in intri.items()}
            with open(stored_path, "w") as f:
                json.dump(intri_to_store, f)
            logger.info(f"Stored Exo intrinsics in {stored_path}.")
        if self.take_data["ego_cam"] in intri:
            intri.pop(self.take_data["ego_cam"])
        return intri

    def load_egopose_annotations(
        self, ann_type="body", smooth_egopose=False, infill_missing=True, load_automatic=False
    ):
        """
        Load ego pose annotations for a take with other helper utilities for infilling small gaps
        and smoothing. This was NOT used for EE4D-Motion data generation pipeline because the
        these annotations were too sparse and too jittery.
        """
        assert ann_type in ["body", "hand"], f"Invalid annotation type {ann_type}."
        take_uid = self.take_data["take_uid"]
        take_name = self.take_data["take_name"]
        if take_uid not in self.egopose_uid2split:
            logger.warning(f"Egopose annotations doesn't exist for {take_name} {take_uid}.")
            return None
        if self.take_data["parent_task_name"] == "Rock Climbing" and ann_type == "body":
            logger.info(f"Discarding egopose body annotations for {take_name} {take_uid}.")
            return None

        def load_annotation(_ann_dir):
            with open(_ann_dir) as f:
                annotation = json.load(f)
            # Remove frames with no annotations
            for frame in list(annotation.keys()):
                if len(annotation[frame]) == 0:
                    annotation.pop(frame)
                elif len(annotation[frame][0]["annotation3D"]) == 0 and len(annotation[frame][0]["annotation2D"]) == 0:
                    annotation.pop(frame)
            return annotation

        # Load manual annotations
        _ann_dir = f"{self.data_dir}/annotations/ego_pose/{self.egopose_uid2split[take_uid]}/{ann_type}/annotation/{take_uid}.json"
        if not os.path.exists(_ann_dir):
            return None
        annotation = load_annotation(_ann_dir)
        manual_ann_frames = [int(frame) for frame in annotation.keys()]

        # Load automatic annotations
        if load_automatic:
            _ann_dir = f"{self.data_dir}/annotations/ego_pose/{self.egopose_uid2split[take_uid]}/{ann_type}/automatic/{take_uid}.json"
            if os.path.exists(_ann_dir):
                annotation_30fps = load_annotation(_ann_dir)
                for frame in annotation_30fps.keys():
                    if frame not in annotation:
                        annotation[frame] = annotation_30fps[frame]
            else:
                logger.warning(f"No automatic egopose annotations found for {take_name} {take_uid}.")

        ann_frames = [int(frame) for frame in annotation.keys()]
        joint_names = BODY_JOINT_NAMES if ann_type == "body" else HAND_JOINTS_NAME

        # Get 3D keypoints
        kp3d = {}
        for frame in ann_frames:
            frame_kp3d = annotation[str(frame)][0]["annotation3D"]
            if len(frame_kp3d):
                kp3d[frame] = parse_kp3d(frame_kp3d, joint_names)

        # Get 2D keypoints
        kp2d = {}
        for frame in ann_frames:
            frame_kp2d = annotation[str(frame)][0]["annotation2D"]
            for cam in frame_kp2d.keys():
                if len(frame_kp2d[cam]):
                    if cam not in kp2d:
                        kp2d[cam] = {}
                    kp2d[cam][frame] = parse_kp2d(frame_kp2d[cam], joint_names)

        if smooth_egopose:
            for cam in list(kp2d.keys()):
                kp2d[cam] = smooth_egopose_annotation(kp2d[cam])
            kp3d = smooth_egopose_annotation(kp3d)

        if infill_missing:
            for cam in list(kp2d.keys()):
                kp2d[cam] = egopose_infill_missing(kp2d[cam])
            kp3d = egopose_infill_missing(kp3d)

        egopose_ann = {"kp3d": kp3d, "kp2d": kp2d, "manual_ann_frames": manual_ann_frames}
        return egopose_ann

    def get_pointcloud_path(self):
        capture_name = self.metadata[self.take_data["take_name"]]["capture"]["capture_name"]
        # first check if pointcloud exists in the capture directory
        pc_path = f"{self.data_dir}/captures/{capture_name}/trajectory/semidense_points.csv.gz"
        if os.path.exists(pc_path):
            return pc_path

        # now check if it exists in any of the take directories
        # because take and capture pointclouds are the same
        for take_name in [self.take_data["take_name"]] + list(self.metadata.keys()):
            if self.metadata[take_name]["capture"]["capture_name"] == capture_name:
                pc_path = f"{self.data_dir}/takes/{take_name}/trajectory/semidense_points.csv.gz"
                if os.path.exists(pc_path):
                    return pc_path

        # if not found, return capture pointcloud path
        pc_path = f"{self.data_dir}/captures/{capture_name}/trajectory/semidense_points.csv.gz"
        return pc_path


if __name__ == "__main__":

    data_dir = "/vision/u/chpatel/data/egoexo4d_ee4d_motion"
    dataset = EgoExo4D_Take_Dataset(data_dir, lowres=True)

    take_name = "cmu_bike02_4"
    take = dataset.load_take(take_name)

    for imgs_data in dataset.get_rgb_generator():  # process frame-wise data sequentially
        break

    import IPython

    IPython.embed()
