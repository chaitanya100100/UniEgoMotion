import os
import IPython
import numpy as np
import torch
import copy
import pytorch_lightning as pl
from tqdm.auto import tqdm


from loguru import logger
from torch.utils.data import Dataset, DataLoader

from dataset.canonicalization import get_a_canonicalized_segment
from dataset.representation_utils import repre_to_full_sequence, saved_sequence_to_repre
from dataset.smpl_utils import get_smpl, evaluate_smpl
from utils.vis_utils import save_video, visualize_sequence, visualize_sequence_blender
from utils.torch_utils import careful_collate_fn
from dataset.feats import ImageFeats


class EE4D_Motion_Dataset(Dataset):
    def __init__(
        self,
        *,
        data_dir,
        split,
        repre_type,
        cond_traj,
        cond_img_feat,
        cond_betas,
        window,
        img_feat_type,
        do_normalization=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.cond_img_feat = cond_img_feat
        self.cond_traj = cond_traj
        self.repre_type = repre_type
        self.img_feat_type = img_feat_type
        self.cond_betas = cond_betas
        self.dataset_name = "ee4d"
        if self.cond_betas:
            logger.warning("Conditioning on betas.")
        assert self.split in ["train", "val"]
        assert self.img_feat_type in ["clip_all", "egovideo", "dinov2", "dinov2_reg"]

        if not self.cond_traj:
            logger.warning("NOT USING TRAJ AS CONDITION.")

        self.window = window
        self.segment_stride = 20
        self.do_normalization = do_normalization
        if not self.do_normalization:
            logger.warning("NOT NORMALIZING. PLEASE CHECK.")

        # Load SMPL
        self.smpl = get_smpl()

        self.load_motion_data()
        self.load_statistics()
        if self.cond_img_feat:
            self.img_feats = ImageFeats(self.data_dir, self.split, self.img_feat_type)

    def load_motion_data(self):
        processed_path = f"{self.data_dir}/uniegomotion/ee_{self.split}.pt"

        logger.info(f"Loading {processed_path}.")
        self.motion_data = torch.load(processed_path, weights_only=False)

        self.seq_names = list(self.motion_data.keys())
        logger.info(f"Loaded {len(self.motion_data)} sequences.")

        # Create a mapping from index to (seq_idx, frame_idx)
        # We use a stride of 20 frames to create segments.
        self.idx_to_sidx_fidx = []
        for seq_idx, seq_name in enumerate(self.seq_names):
            T = self.motion_data[seq_name]["num_frames"]
            for fidx in range(0, T - self.segment_stride, self.segment_stride):
                self.idx_to_sidx_fidx.append((seq_idx, fidx))
        self.idx_to_sidx_fidx = np.array(self.idx_to_sidx_fidx).astype(int)

    def load_statistics(self):
        if not self.do_normalization:
            return
        logger.warning("Loading stats for the window of 80 frames.")
        self.stats = torch.load(f"{self.data_dir}/uniegomotion/{self.repre_type}_ee_train_stats.pt", weights_only=False)
        clean_it = lambda x: torch.where(x.abs() < 1e-8, torch.ones_like(x), x)
        self.stats["traj_std"] = clean_it(self.stats["traj_std"])
        self.stats["motion_std"] = clean_it(self.stats["motion_std"])

    def __len__(self):
        return len(self.idx_to_sidx_fidx)

    def normalize(self, a, k):
        if not self.do_normalization:
            return a
        na = (a - self.stats[k + "_mean"]) / (self.stats[k + "_std"] + 1e-6)
        return na

    def denormalize(self, a, k):
        if not self.do_normalization:
            return a
        na = a * (self.stats[k + "_std"] + 1e-6) + self.stats[k + "_mean"]
        return na

    def pad_to_window(self, ret):
        valid_frames = torch.ones(self.window).long()
        T = ret["misc"]["motion"].shape[0]
        if T != self.window:
            valid_frames[T:] = 0  # mark invalid frames
            pad_fn = lambda x: torch.cat([x, torch.zeros(self.window - T, *x.shape[1:], dtype=x.dtype)], dim=0)

            if "traj" in ret["y"]:
                ret["y"]["traj"] = pad_fn(ret["y"]["traj"])
            if "img_embs" in ret["y"]:
                ret["y"]["img_embs"] = pad_fn(ret["y"]["img_embs"])
                ret["y"]["valid_img_embs"] = pad_fn(ret["y"]["valid_img_embs"])
            if "traj" in ret["misc"]:
                ret["misc"]["traj"] = pad_fn(ret["misc"]["traj"])
            if "motion" in ret["misc"]:
                ret["misc"]["motion"] = pad_fn(ret["misc"]["motion"])

        ret["y"]["valid_frames"] = valid_frames  # determines the number of frames
        return ret

    def __getitem__(self, idx):
        sidx, st = self.idx_to_sidx_fidx[idx]
        seq_name = self.seq_names[sidx]
        return self.get_from_seq_and_st(seq_name, st, idx)

    def get_from_seq_and_st(self, seq_name, st, idx):
        en = min(st + self.window - 1, self.motion_data[seq_name]["num_frames"] - 1)
        floor_height = self.motion_data[seq_name]["floor_height"]
        # logger.warning(f"Sequence: {seq_name}, start index: {st}, end index: {en}")

        segment_data = get_a_canonicalized_segment(
            self.motion_data[seq_name]["smpl_params"],
            self.motion_data[seq_name]["aria_traj"],
            self.motion_data[seq_name]["kp3d"],
            self.smpl,
            st,
            en,
        )

        motion, traj = saved_sequence_to_repre(
            self.repre_type,
            segment_data["can_aria_traj"],
            segment_data["can_smpl_params"],
            segment_data["can_kp3d"],
            floor_height,
            self.smpl,
        )

        motion = self.normalize(motion, "motion")
        traj = self.normalize(traj, "traj")
        if self.cond_img_feat:
            img_embs, valid_img_embs = self.img_feats.get_img_feats(
                seq_name, segment_data["start_idx"], segment_data["end_idx"]
            )  # T x 768

        if self.cond_betas:
            betas = segment_data["can_smpl_params"]["betas"][:1].view(10)

        ret = {}
        ret["y"] = {}  # holds conditioning inputs
        ret["pred"] = {}  # holds predicted outputs
        if self.cond_traj:
            ret["y"]["traj"] = traj
        if self.cond_img_feat:
            ret["y"]["img_embs"] = img_embs
            ret["y"]["valid_img_embs"] = valid_img_embs

        if self.cond_betas:
            ret["y"]["betas"] = betas
        ret["misc"] = {  # holds ground truth and other metadata
            "seq_name": seq_name,
            "start_idx": segment_data["start_idx"],
            "end_idx": segment_data["end_idx"],
            "idx": idx,
            "dataset": self.dataset_name,
            "traj": traj,
            "motion": motion,
        }
        ret = self.pad_to_window(ret)
        return ret

    def ret_to_full_sequence(self, ret):
        is_batch = True
        if not isinstance(ret["misc"]["seq_name"], list):
            ret = careful_collate_fn([ret])
            is_batch = False

        mdata = {
            "aria_traj_T": [],
            "smpl_params_full": [],
            "kp3d": [],
            "verts": [],
            "full_pose": [],
        }

        assert "motion" not in ret and "traj" not in ret

        for i in range(len(ret["misc"]["seq_name"])):
            vf = ret["y"]["valid_frames"][i].bool()  # T_window
            # If prediction is available, use it. Otherwise, use ground truth.
            motion = ret["pred"]["motion"] if "motion" in ret["pred"] else ret["misc"]["motion"]
            traj = ret["pred"]["traj"] if "traj" in ret["pred"] else ret["misc"]["traj"]
            motion = self.denormalize(motion[i][vf], "motion")
            traj = self.denormalize(traj[i][vf], "traj")

            seq_name = ret["misc"]["seq_name"][i]

            betas = self.motion_data[seq_name]["smpl_params"]["betas"]  # 1 x 10
            body_root_offset = self.motion_data[seq_name]["body_root_offset"]
            aria_traj_T, smpl_params_full, _ = repre_to_full_sequence(
                self.repre_type, motion, traj, self.smpl, betas, body_root_offset
            )
            kp3d, verts, full_pose = evaluate_smpl(self.smpl, smpl_params_full)

            mdata["aria_traj_T"].append(aria_traj_T)
            mdata["smpl_params_full"].append(smpl_params_full)
            mdata["kp3d"].append(kp3d)
            mdata["verts"].append(verts)
            mdata["full_pose"].append(full_pose)

        return mdata if is_batch else {k: v[0] for k, v in mdata.items()}

    def process_sample_for_task(self, ret, task):
        if task == "recon":
            return ret
        if task == "gen":
            ret = copy.deepcopy(ret)
            ret["y"]["traj_mask"] = torch.ones(self.window).long()  # T
            ret["y"]["img_mask"] = torch.ones(self.window).long()  # T
            ret["y"]["img_mask"][0] = 0  # first frame is visible, everything else is masked out.
            ret["y"]["valid_frames"] = torch.ones_like(ret["y"]["valid_frames"])  # generate full T frames
            return ret
        if task in ["fore"]:
            ret = copy.deepcopy(ret)
            ret["y"]["traj_mask"] = torch.ones(self.window).long()  # T
            ret["y"]["img_mask"] = torch.ones(self.window).long()  # T

            avail = min(self.window // 4, ret["y"]["valid_frames"].sum())
            ret["y"]["traj_mask"][:avail] = 0  # 20 frames are visible
            ret["y"]["img_mask"][:avail] = 0  # 20 frames are visible
            ret["y"]["valid_frames"] = torch.ones_like(ret["y"]["valid_frames"])  # generate full T frames

            return ret

    def visualize_sample(self, ret, use_blender=False):

        is_batch = True
        if not isinstance(ret["misc"]["seq_name"], list):
            ret = careful_collate_fn([ret])
            is_batch = False

        is_pred_traj, is_pred_motion = "traj" in ret["pred"], "motion" in ret["pred"]
        pred_mdata = self.ret_to_full_sequence(ret)

        gt_ret = copy.deepcopy(ret)
        gt_ret["pred"] = {}
        gt_mdata = self.ret_to_full_sequence(gt_ret)

        vis = []
        for i in range(len(ret["misc"]["seq_name"])):
            vis_fn = visualize_sequence if not use_blender else visualize_sequence_blender
            imgs = vis_fn(
                aria_traj=gt_mdata["aria_traj_T"][i],
                verts=gt_mdata["verts"][i],
                # global_jpos=gt_mdata["kp3d"][i],
                pred_aria_traj=pred_mdata["aria_traj_T"][i] if is_pred_traj else None,
                pred_verts=pred_mdata["verts"][i] if is_pred_motion else None,
                faces=self.smpl.faces,
            )
            vis.append(imgs)

        return vis if is_batch else vis[0]


class EE4D_Motion_DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str):
        assert stage in ["fit", "validate", "test", "predict", "train", "val"]
        kwargs = dict(
            data_dir=self.cfg.DATA.DATA_DIR,
            repre_type=self.cfg.DATA.REPRE_TYPE,
            cond_img_feat=self.cfg.DATA.COND_IMG_FEAT,
            cond_traj=self.cfg.DATA.COND_TRAJ,
            window=self.cfg.DATA.WINDOW,
            img_feat_type=self.cfg.DATA.IMG_FEAT_TYPE,
            cond_betas=self.cfg.DATA.COND_BETAS,
        )
        dataset_name = self.cfg.DATA.DATASET_NAME
        assert dataset_name in ["ee4d"]

        if stage in ["fit", "train"]:
            self.train_dataset = EE4D_Motion_Dataset(split="train", **kwargs)
            logger.info(f"Train dataset: {len(self.train_dataset)}")

        self.val_dataset = EE4D_Motion_Dataset(split="val", **kwargs)
        logger.info(f"Val dataset: {len(self.val_dataset)}")

    def val_dataloader(self, shuffle=False):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.DATA.BATCH_SIZE,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            shuffle=shuffle,
            collate_fn=careful_collate_fn,
            # pin_memory=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.DATA.BATCH_SIZE,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            shuffle=True,
            collate_fn=careful_collate_fn,
            # pin_memory=False,
        )


def save_statistics():
    repre_type = "v5_beta"
    window = 80  # This is the window size used in the paper.
    data_dir = "/vision/u/chpatel/data/egoexo4d_ee4d_motion"
    out_file = f"{data_dir}/uniegomotion/{repre_type}_ee_train_stats.pt"

    if os.path.exists(out_file):
        logger.info(f"{out_file} exists. Skipping.")
        return

    dataset = EE4D_Motion_Dataset(
        data_dir=data_dir,
        split="train",
        repre_type=repre_type,
        cond_traj=True,
        cond_img_feat=False,
        window=window,
        img_feat_type="clip_all",
        do_normalization=False,
        cond_betas=False,
    )

    all_motion = []
    all_traj = []
    for sample in tqdm(dataset):
        vf = sample["y"]["valid_frames"].bool()
        motion = sample["misc"]["motion"][vf]
        traj = sample["misc"]["traj"][vf]
        all_motion.append(motion)
        all_traj.append(traj)
    all_motion = torch.cat(all_motion, dim=0)  # T_dataset x F
    all_traj = torch.cat(all_traj, dim=0)  # T_dataset x F

    stats = {}
    stats["motion_mean"] = all_motion.mean(dim=0)
    stats["motion_std"] = all_motion.std(dim=0)
    stats["motion_min"] = all_motion.min(dim=0).values
    stats["motion_max"] = all_motion.max(dim=0).values

    stats["traj_mean"] = all_traj.mean(dim=0)
    stats["traj_std"] = all_traj.std(dim=0)
    stats["traj_min"] = all_traj.min(dim=0).values
    stats["traj_max"] = all_traj.max(dim=0).values

    torch.save(stats, out_file)


def vis_a_sample():
    data_dir = "/vision/u/chpatel/data/egoexo4d_ee4d_motion"
    out_dir = "/vision/u/chpatel/test"
    window = 80
    use_blender = False  # use_blender=True for blendify blender visualization if available.

    os.makedirs(out_dir, exist_ok=True)

    dataset = EE4D_Motion_Dataset(
        data_dir=data_dir,
        window=window,
        split="val",
        # The following args are not important for visualization.
        repre_type="v1",
        cond_traj=True,
        cond_img_feat=False,
        cond_betas=False,
        img_feat_type="dinov2",
        do_normalization=False,
    )

    # After processing, each take is broken down into several 'good' segments, discarding segments with bad optim results.
    # Thus each sequence name is '<take_name>___<start_frame_index>___<end_frame_index>' where frame indices are according
    # to 30fps numbering and inclusive. Note that EE4D Motion dataset is at 10 fps. For example,
    # uniandes_basketball_003_42___585___825 sequence will have smpl motion sequence of length (825 - 585) / 3 + 1 = 81 at 10fps.

    # You can print all sequence names for that particular split
    # print(dataset.seq_names)

    # Choose a sequence to visualize
    seq = "iiith_cooking_58_2___2478___3498"
    seq = "uniandes_basketball_003_42___585___825"

    # Choose start index within that segment. It will visualize the segment [st:st+window] within `seq`
    st = 0

    sample = dataset.get_from_seq_and_st(seq, st, idx=0)  # idx=0 is for bookkeeping during training. Ignore it here.
    imgs = dataset.visualize_sample(sample, use_blender=use_blender)
    save_video(imgs[..., ::-1], f"{seq}___starting_{st}", out_dir, fps=10)

    import IPython

    IPython.embed()


if __name__ == "__main__":

    save_statistics()
    # vis_a_sample()

    import IPython

    IPython.embed()
