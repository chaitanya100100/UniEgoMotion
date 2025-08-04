import numpy as np
import joblib
from loguru import logger
import copy
import os
import cv2
import argparse

from dataset.egoexo4d_take_dataset import EgoExo4D_Take_Dataset
from utils.renderer import Renderer
from utils.torch_utils import to_tensor, dcn
from dataset.smpl_utils import get_smpl, evaluate_smpl
from utils.vis_utils import put_text_on_img
from utils import rotation_conversions as rc
from utils.pca_conversions import pca_to_matrix, matrix_to_pca


VIZ_SCALE = 0.3


def project_torch(pred_kp3d, intri, extri):
    # Project 3D keypoints to 2D using camera intrinsics and extrinsics.
    # pred_kp3d: J x 3 torch tensor
    # intri: 3 x 3 torch tensor
    # extri: 3 x 4 torch tensor

    cam_kp3d = (extri[:, :3] @ pred_kp3d.T).T + extri[:, 3]  # J x 3
    cam_kp2d = (intri @ cam_kp3d.T).T  # J x 3
    cam_kp2d = cam_kp2d[:, :2] / (cam_kp2d[:, 2:] + 1.0e-9)  # J x 2
    return cam_kp2d


def forward_pass(smpl, smpl_params, intri, extri):
    smpl_input = {k: v for k, v in smpl_params.items()}  # new dict

    # Convert SMPL parameters to the format expected by the SMPL model
    # Rotations are stored as 6D rotations and handpose as PCA coefficients.
    # Note that hand pose parameters were NOT fitted in EE4D-Motion dataset.
    smpl_input["body_pose"] = rc.rotation_6d_to_matrix(smpl_input["body_pose"])
    smpl_input["global_orient"] = rc.rotation_6d_to_matrix(smpl_input["global_orient"])
    if "left_hand_pose" in smpl_input:
        smpl_input["left_hand_pose"] = pca_to_matrix(smpl_input["left_hand_pose"], smpl.left_hand_components)
    if "right_hand_pose" in smpl_input:
        smpl_input["right_hand_pose"] = pca_to_matrix(smpl_input["right_hand_pose"], smpl.right_hand_components)
    smpl_input = {k: v[None] for k, v in smpl_input.items()}

    smpl_output = smpl.forward(**smpl_input)
    pred_vertices = smpl_output.vertices[0]  # V x 3
    pred_kp3d = smpl_output.joints[0]  # J' x 3
    pred_kp2d = {cam: project_torch(pred_kp3d, intri[cam], extri[cam]) for cam in intri.keys()}
    return pred_vertices, pred_kp3d, pred_kp2d


def viz_smpl(smpl_params, smpl, take, imgs_data, renderer):
    frame_idx = imgs_data["frame_idx"]
    # Moving aria camera has different extrinsics for each frame whereas static exo cameras have fixed extrinsics.
    extri = {cam: v if "aria" not in cam.lower() else v[frame_idx] for cam, v in take["extri"].items()}

    smpl_params = {k: v.detach() for k, v in smpl_params.items()}
    pred_vertices, pred_kp3d, pred_kp2d = forward_pass(smpl, smpl_params, extri=extri, intri=imgs_data["intri"])

    ret = {}
    for cam in pred_kp2d.keys():
        inp_img = dcn(imgs_data["rgb"][cam] * 255).astype(np.uint8)
        intri_cam = dcn(imgs_data["intri"][cam])
        extri_cam = dcn(extri[cam])

        # Scale by VIZ_SCALE for rendering. Change intrinsics accordingly.
        sc = VIZ_SCALE if "aria" not in cam.lower() else 1.0
        intri_cam = intri_cam.copy()
        intri_cam[:2, :3] = intri_cam[:2, :3] * sc
        inp_img = cv2.resize(inp_img, (0, 0), fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)

        vis_img = renderer.image_render(
            dcn(pred_vertices),
            smpl.faces,
            image=inp_img,
            camera_intrinsics=intri_cam,
            camera_extrinsics=extri_cam,
            alpha=0.75,
        )

        # Undistorted aria cameras are rotated by 90 degrees. This aligns with its extrinsics and intrinsics.
        # However, for visualization, we rotate the image back to upright.
        if "aria" in cam.lower():
            vis_img = np.rot90(vis_img, -1).copy()

        ret[cam] = vis_img
        ret[cam] = put_text_on_img(ret[cam], f"{frame_idx}")
    return ret


def main():
    parser = argparse.ArgumentParser(description="Visualize SMPL motion results.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--take_name", type=str, required=True, help="Name of the take to visualize")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for visualizations")
    args = parser.parse_args()

    data_dir = args.data_dir
    take_name = args.take_name
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=False)

    dataset = EgoExo4D_Take_Dataset(data_dir, lowres=True)
    take = dataset.load_take(take_name)
    take = to_tensor(take)

    # Load optim results
    optim_results = joblib.load(f"{data_dir}/ee4d_motion/{take_name}/multiview_optim_2.pkl")
    optim_results = to_tensor(optim_results)

    # Load SMPL model
    smpl = get_smpl()

    # Renderer
    renderer = Renderer()

    # Select ego camera and 2 exo cameras for visualization
    all_cams = [take["ego_cam"], take["exo_cams"][0], take["exo_cams"][1]]
    best_exo = dataset.metadata[take_name]["best_exo"]
    if best_exo not in all_cams:
        all_cams[2] = best_exo

    # Iterate through frames and visualize
    for imgs_data in dataset.get_rgb_generator(num_workers=0, all_cams=all_cams):
        imgs_data = to_tensor(imgs_data)
        frame_idx = imgs_data["frame_idx"]

        log_info = f"{take_name} {frame_idx}"
        logger.info(log_info)

        # Blank visualization in case of missing data
        blank_viz = {}
        for cam in all_cams:
            temp = dcn(imgs_data["rgb"][cam] * 255).astype(np.uint8)
            temp = temp if "aria" not in cam.lower() else np.rot90(temp, -1).copy()
            sc = VIZ_SCALE if "aria" not in cam.lower() else 1.0
            blank_viz[cam] = cv2.resize(temp, (0, 0), fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
            blank_viz[cam] = put_text_on_img(blank_viz[cam], f"{frame_idx}")

        # Function to get visualization for a specific set of smpl parameters
        def get_good_viz(key):
            if frame_idx not in optim_results or key not in optim_results[frame_idx]:
                logger.warning(f"{log_info}. {key} not found in optimization results.")
                return copy.deepcopy(blank_viz)
            else:
                smpl_params = optim_results[frame_idx][key]
                return viz_smpl(smpl_params, smpl, take, imgs_data, renderer)

        # init_viz = get_good_viz("init_smpl_params")
        # tfix_viz = get_good_viz("tfix_smpl_params")
        final_viz = get_good_viz("final_smpl_params")

        # Save visualizations
        for cam in final_viz.keys():
            cv2.imwrite(f"{out_dir}/{cam}_{frame_idx:06d}.jpg", final_viz[cam][..., ::-1])

    for cam in all_cams:
        fps = dataset.fps
        cmd = f"ffmpeg -y -framerate {fps} -pattern_type glob -i '{out_dir}/{cam}_*.jpg' -c:v libx264 -pix_fmt yuv420p {out_dir}/{cam}.mp4"
        print(cmd)
        os.system(cmd)

        # delete images
        os.sync()
        cmd = f"rm -rf {out_dir}/{cam}_*.jpg"
        os.system(cmd)


if __name__ == "__main__":
    main()
