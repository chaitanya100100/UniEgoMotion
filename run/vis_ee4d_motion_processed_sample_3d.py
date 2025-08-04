import torch
import os

from utils.vis_utils import save_video, visualize_sequence, visualize_sequence_blender
from dataset.canonicalization import get_a_canonicalized_segment
from dataset.smpl_utils import get_smpl, evaluate_smpl


def main():
    # put your motion data path here.
    motion_data_path = "/vision/u/chpatel/data/egoexo4d_ee4d_motion/uniegomotion/ee_val.pt"
    out_dir = "/vision/u/chpatel/test"
    use_blender = True  # use_blender=True for blendify blender visualization if available.

    os.makedirs(out_dir, exist_ok=True)

    motion_data = torch.load(motion_data_path, weights_only=False)
    smpl = get_smpl()

    # After processing, each take is broken down into several 'good' segments, discarding segments with bad optim results.
    # Thus each sequence name is '<take_name>___<start_frame_index>___<end_frame_index>' where frame indices are according
    # to 30fps numbering and inclusive. Note that EE4D Motion dataset is at 10 fps. For example,
    # uniandes_basketball_003_42___585___825 sequence will have smpl motion sequence of length (825 - 585) / 3 + 1 = 81 at 10fps.

    # You can print all sequence names for that particular split
    # print(motion_data.keys())

    # Choose a sequence to visualize
    seq = "iiith_cooking_58_2___2478___3498"
    seq = "uniandes_basketball_003_42___585___825"

    # Choose a sub-sequence to visualize
    start_index = 0
    end_index = 80  # This is the window size used in the paper.

    # Transform such that first frame is facing -Y and at the origin of XY plane. Z is up.
    sample = get_a_canonicalized_segment(
        smpl_params=motion_data[seq]["smpl_params"],
        aria_traj=motion_data[seq]["aria_traj"],
        kp3d=motion_data[seq]["kp3d"],
        smpl=smpl,
        start_idx=start_index,
        end_idx=end_index,
    )

    _, can_verts, _ = evaluate_smpl(smpl, sample["can_smpl_params"])

    vis_fn = visualize_sequence if not use_blender else visualize_sequence_blender
    imgs = vis_fn(
        aria_traj=sample["can_aria_traj"],
        verts=can_verts,
        faces=smpl.faces,
        floor_height=motion_data[seq]["floor_height"],
    )

    save_video(imgs[..., ::-1], f"{seq}___starting_{start_index}", out_dir, fps=10)

    import IPython

    IPython.embed()


if __name__ == "__main__":
    main()
