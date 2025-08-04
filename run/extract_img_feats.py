import argparse
import os
import numpy as np
import torch
from loguru import logger
import joblib
import clip
from PIL import Image
from tqdm.auto import tqdm
import json


from dataset.egoexo4d_take_dataset import EgoExo4D_Take_Dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/vision/u/chpatel/data/egoexo4d_ee4d_motion")
    parser.add_argument("--out_dir", type=str, default="/vision/group/egoexo4d/ours")
    parser.add_argument("--take_name", type=str, help="Name of the take to be renderd.")
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "run_clip_ego",
            "run_dinov2_ego",
            "merge_clip_ego",
            "merge_dinov2_ego",
            "merge_egovideo_ego",
        ],
        help="Task to run.",
    )
    parser.add_argument("--clip_model", type=str, default="ViT-L/14@336px", help="CLIP model to use.")
    parser.add_argument("--dinov2_model", type=str, default="dinov2_vitl14_reg", help="DINOv2 model to use.")
    parser.add_argument("--jit", action="store_true", help="Use jit.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--split", type=str, default="train", help="Split to use for merging.")

    return parser.parse_args()


"""
['RN50',
 'RN101',
 'RN50x4',
 'RN50x16',
 'RN50x64',
 'ViT-B/32',
 'ViT-B/16',
 'ViT-L/14',
 'ViT-L/14@336px']
"""


def run_clip(do_ego=False):
    args = get_args()
    logger.info(args)

    # Create output directory if it doesn't exist
    out_dir = f"{args.out_dir}/{args.take_name}"
    os.makedirs(out_dir, exist_ok=True)

    # output file
    out_file = f"{out_dir}/clip_image.pkl" if not do_ego else f"{out_dir}/clip_image_ego.pkl"
    if os.path.exists(out_file) and not args.overwrite:
        logger.info(f"{out_file} exists. Skipping.")
        return

    # Load dataset
    dataset = EgoExo4D_Take_Dataset(args.data_dir, lowres=False)
    take = dataset.load_take(args.take_name)

    # load model
    device = torch.device("cuda")
    clip_model, clip_process = clip.load(args.clip_model, device=device, jit=args.jit)
    clip_model.eval()

    cams_todo = take["exo_cams"] if not do_ego else [take["ego_cam"]]
    out_dict = {cam: {} for cam in cams_todo}

    for cam in cams_todo:
        for imgs_data in dataset.get_rgb_generator(all_cams=[cam], num_workers=0, local_copy=True):
            frame_idx = imgs_data["frame_idx"]
            rgb = imgs_data["rgb"][cam]
            # if ego, make image upright
            if cam == take["ego_cam"]:
                rgb = np.rot90(rgb, -1).copy()
            rgb = Image.fromarray((rgb * 255).astype(np.uint8))
            rgb = clip_process(rgb)

            with torch.inference_mode():
                feats = clip_model.encode_image(rgb[None].to(device))[0]
                feats = feats.cpu().numpy()

            out_dict[cam][frame_idx] = feats
            log_info = f"{args.take_name} {cam} {frame_idx}"
            logger.info(f"{log_info}.")

    joblib.dump(out_dict, out_file)
    logger.info(f"Saved to {out_file}")


def run_dinov2(do_ego=False, img_size=336, fps=10, save_all_prenorm=False):
    args = get_args()
    logger.info(args)

    # Create output directory if it doesn't exist
    out_dir = f"{args.out_dir}/{args.take_name}"
    os.makedirs(out_dir, exist_ok=True)

    # output file
    out_file = f"{out_dir}/dinov2_image.pkl" if not do_ego else f"{out_dir}/dinov2_image_ego.pkl"
    if save_all_prenorm:
        out_file = out_file.replace(".pkl", f"_all_prenorm_{fps}fps_im{img_size}.pkl")
    if os.path.exists(out_file) and not args.overwrite:
        logger.info(f"{out_file} exists. Skipping.")
        return

    # Load dataset
    orig_fps = 30
    dataset = EgoExo4D_Take_Dataset(args.data_dir, lowres=False, fps=orig_fps)
    take = dataset.load_take(args.take_name)

    # Change fps
    # fps = 5
    assert orig_fps % fps == 0
    dataset.fps = fps
    take["frames"] = take["frames"][:: orig_fps // fps]
    logger.info(f"Using frames {take['frames'][:100]}...")

    # Transform copied from CLIP
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    def _transform(n_px):
        import torchvision.transforms as T

        try:
            BICUBIC = T.InterpolationMode.BICUBIC
        except ImportError:
            BICUBIC = Image.BICUBIC

        return T.Compose(
            [
                T.Resize(n_px, interpolation=BICUBIC),
                # CenterCrop(n_px),
                _convert_image_to_rgb,
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    if save_all_prenorm:
        assert args.dinov2_model == "dinov2_vitb14_reg"

    # load model
    device = torch.device("cuda")
    dinov2_model = torch.hub.load("facebookresearch/dinov2", args.dinov2_model)
    dinov2_model.to(device).eval()
    dinov2_process = _transform(img_size)

    cams_todo = take["exo_cams"] if not do_ego else [take["ego_cam"]]
    out_dict = {cam: {} for cam in cams_todo}

    for cam in cams_todo:
        for imgs_data in dataset.get_rgb_generator(all_cams=[cam], num_workers=0, local_copy=True):
            frame_idx = imgs_data["frame_idx"]
            rgb = imgs_data["rgb"][cam]
            # if ego, make image upright
            if cam == take["ego_cam"]:
                rgb = np.rot90(rgb, -1).copy()
            rgb = Image.fromarray((rgb * 255).astype(np.uint8))
            rgb = dinov2_process(rgb)

            with torch.inference_mode():
                feats = dinov2_model.forward_features(rgb[None].to(device))
                if save_all_prenorm:
                    feats = feats["x_prenorm"]
                else:
                    # one cls token and 4 reg tokens
                    feats = torch.cat([feats["x_norm_clstoken"][0][None], feats["x_norm_regtokens"][0]], 0)
                feats = feats.to(torch.float16).cpu().numpy()
                # assert feats.shape[0] == 5

            out_dict[cam][frame_idx] = feats
            log_info = f"{args.take_name} {cam} {frame_idx}"
            logger.info(f"{log_info}.")

    joblib.dump(out_dict, out_file)
    logger.info(f"Saved to {out_file}")


def merge_clip_ego():
    args = get_args()
    logger.info(args)

    # output file
    split = args.split
    out_file = f"{args.data_dir}/uniegomotion/egoview_clip_{split}.pt"
    if os.path.exists(out_file) and not args.overwrite:
        logger.info(f"{out_file} exists. Skipping.")
        return

    with open(f"{args.data_dir}/annotations/splits.json", "r") as f:
        splits = json.load(f)["split_to_take_uids"]

    dataset = EgoExo4D_Take_Dataset(args.data_dir, lowres=False)

    out_dict = {
        "feats": {},
        "egocam": {},
    }
    for take_uid in tqdm(splits[split]):
        if take_uid not in dataset.uid2name:
            continue

        take_name = dataset.uid2name[take_uid]
        ego_clip_path = f"{args.out_dir}/{take_name}/clip_image_ego.pkl"

        if not os.path.exists(ego_clip_path):
            logger.info(f"{ego_clip_path} does not exist. Skipping.")
            continue

        dct = joblib.load(ego_clip_path)
        assert len(dct.values()) == 1
        out_dict["egocam"][take_name] = list(dct.keys())[0]
        dct = list(dct.values())[0]

        frame_idxs = np.array(sorted(dct.keys()))
        assert frame_idxs[0] == 0, frame_idxs
        assert np.all(frame_idxs == np.arange(0, frame_idxs[-1] + 1, 3))

        feats = torch.from_numpy(np.stack([dct[frame_idx] for frame_idx in frame_idxs]))
        out_dict["feats"][take_name] = feats

    torch.save(out_dict, out_file)
    logger.info(f"Saved to {out_file}")


def merge_dinov2_ego():
    args = get_args()
    logger.info(args)

    # output file
    split = args.split
    out_file = f"{args.data_dir}/uniegomotion/egoview_dinov2_{split}.pt"
    if os.path.exists(out_file) and not args.overwrite:
        logger.info(f"{out_file} exists. Skipping.")
        return

    with open(f"{args.data_dir}/annotations/splits.json", "r") as f:
        splits = json.load(f)["split_to_take_uids"]

    dataset = EgoExo4D_Take_Dataset(args.data_dir, lowres=False)

    out_dict = {
        "feats": {},
        "egocam": {},
    }
    for take_uid in tqdm(splits[split]):
        if take_uid not in dataset.uid2name:
            continue

        take_name = dataset.uid2name[take_uid]
        ego_clip_path = f"{args.out_dir}/{take_name}/dinov2_image_ego.pkl"

        if not os.path.exists(ego_clip_path):
            logger.info(f"{ego_clip_path} does not exist. Skipping.")
            continue

        dct = joblib.load(ego_clip_path)
        assert len(dct.values()) == 1
        out_dict["egocam"][take_name] = list(dct.keys())[0]
        dct = list(dct.values())[0]

        frame_idxs = np.array(sorted(dct.keys()))
        assert frame_idxs[0] == 0, frame_idxs
        assert np.all(frame_idxs == np.arange(0, frame_idxs[-1] + 1, 6))

        feats = torch.from_numpy(np.stack([dct[frame_idx] for frame_idx in frame_idxs]))
        out_dict["feats"][take_name] = feats

    torch.save(out_dict, out_file)
    logger.info(f"Saved to {out_file}")


def merge_egovideo_ego():
    args = get_args()
    logger.info(args)

    # output file
    split = args.split
    out_file = f"{args.data_dir}/uniegomotion/egoview_egovideo_{split}.pt"
    if os.path.exists(out_file) and not args.overwrite:
        logger.info(f"{out_file} exists. Skipping.")
        return

    with open(f"{args.data_dir}/annotations/splits.json", "r") as f:
        splits = json.load(f)["split_to_take_uids"]

    dataset = EgoExo4D_Take_Dataset(args.data_dir, lowres=False)

    out_dict = {
        "feats": {},
        "egocam": {},
        "start_frame": {},
        "end_frame": {},
    }
    for take_uid in tqdm(splits[split]):
        if take_uid not in dataset.uid2name:
            continue

        take_name = dataset.uid2name[take_uid]
        ego_clip_path = f"{args.out_dir}/{take_name}/egovideo_feats_ego.pkl"

        if not os.path.exists(ego_clip_path):
            logger.info(f"{ego_clip_path} does not exist. Skipping.")
            continue

        dct = joblib.load(ego_clip_path)
        assert len(dct.values()) == 1
        out_dict["egocam"][take_name] = list(dct.keys())[0]
        dct = list(dct.values())[0]

        vclip_frame_idxs = sorted(dct.keys(), key=lambda x: x[0])
        feats = torch.from_numpy(np.stack([dct[vclip_idxs] for vclip_idxs in vclip_frame_idxs]))
        out_dict["feats"][take_name] = feats

        vclip_frame_idxs = np.array(vclip_frame_idxs)
        out_dict["start_frame"][take_name] = vclip_frame_idxs[:, 0]
        out_dict["end_frame"][take_name] = vclip_frame_idxs[:, -1]

    torch.save(out_dict, out_file)
    logger.info(f"Saved to {out_file}")


if __name__ == "__main__":
    args = get_args()
    if args.task == "run_clip_ego":
        run_clip(do_ego=True)
    elif args.task == "run_dinov2_ego":
        run_dinov2(do_ego=True)
    elif args.task == "merge_clip_ego":
        merge_clip_ego()
    elif args.task == "merge_dinov2_ego":
        merge_dinov2_ego()
    elif args.task == "merge_egovideo_ego":
        merge_egovideo_ego()
    else:
        raise ValueError(f"Task {args.task} not implemented.")
