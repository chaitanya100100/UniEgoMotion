import io
import sys
import warnings

import av
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def read_frames_cv2(video_path, frame_idxs):
    """Read frames from a video file using OpenCV.

    Args:
        video_path (str): path to the video file
        frame_idxs (list): list of frame indices to read
    Returns:
        frames (np array): T x H x W x C np array of frames
        valid: T-length boolean array indicating frame validity
    """
    warnings.warn("cv2 frame reading is not reliable. Use PyAv.")

    frames = []
    valid = []
    cap = cv2.VideoCapture(str(video_path))  # Open the video file
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the number of frames

    for i, fidx in enumerate(frame_idxs):

        # currently keeping this for caution
        assert fidx < vlen, f"Frame index {fidx} exceeds video length {vlen}"

        # Set the frame index and read the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()

        # if the frame is valid, convert it to RGB and append to the list
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            valid.append(1)
        else:
            frames.append(None)
            valid.append(0)

    # currently keeping this for caution
    assert sum(valid) > 0, "No valid frames found"
    assert sum(valid) == len(frames), "Some frames are invalid"

    # Replace invalid frames with zero images
    zero_img = np.zeros_like(frames[valid.index(1)])
    for i, v in enumerate(valid):
        if v == 0:
            frames[i] = zero_img

    # Convert the list of frames to a numpy array
    frames = np.stack(frames).astype(np.float32) / 255
    valid = np.array(valid)
    cap.release()
    return frames, valid


def read_frames_av(video_local_path, frame_idxs):
    # Same as read_frames_cv2 but using PyAv.
    container = av.open(video_local_path)
    frames = [None] * len(frame_idxs)
    frame_count = 0
    for frame in tqdm(container.decode(video=0), desc=f"Reading video frames of {video_local_path}"):
        if frame_count in frame_idxs:
            input_img = np.array(frame.to_image())
            pil_img = Image.fromarray(input_img)
            frames[frame_idxs.index(frame_count)] = pil_img
        frame_count += 1
        if None not in frames:
            break

    valid = [1 if img is not None else 0 for img in frames]
    zero_img = np.zeros_like(frames[valid.index(1)])
    for i, v in enumerate(valid):
        if v == 0:
            frames[i] = zero_img

    assert None not in frames
    frames = np.stack(frames).astype(np.float32) / 255
    valid = np.array(valid)

    return frames, valid


class SequentialVideoReader:
    def __init__(self, video_path, load_in_ram=False):
        self.video_path = video_path
        if load_in_ram:
            with open(video_path, "rb") as file:
                binary_data = file.read()
            print(f"Loaded video file {video_path} into memory (size: {sys.getsizeof(binary_data) / 1e6} megabytes)")
            self.container = av.open(io.BytesIO(binary_data))
        else:
            self.container = av.open(video_path)  # Open the video file
        # self.stream = self.container.streams.video[0]  # Access the video stream
        self.current_frame_index = 0  # To keep track of the current frame index

    def get_frame(self, frame_idx):
        # print("frame_idx", frame_idx, "current_frame_index", self.current_frame_index)

        # Ensure frame_idx is not less than current_frame_index (to maintain sequential reading)
        if frame_idx < self.current_frame_index:
            raise ValueError("frame_idx must be greater than or equal to the current frame index")

        # for frame in self.container.decode(self.stream):
        for frame in self.container.decode(video=0):
            if self.current_frame_index == frame_idx:
                # Convert the frame to a format that can be easily used (e.g., PIL image or numpy array)
                img = frame.to_image()
                self.current_frame_index += 1
                return img
            self.current_frame_index += 1

        raise ValueError("Reached end of video without finding frame_idx")

    def close(self):
        print(f"Closing video file reader {self.video_path}")
        self.container.close()

    def __del__(self):
        self.close()
