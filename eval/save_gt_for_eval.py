import os
import joblib
from tqdm.notebook import tqdm


from utils.torch_utils import careful_collate_fn


def save_params():
    from dataset.ee4d_motion_dataset import EE4D_Motion_Dataset

    data_dir = "/vision/u/chpatel/data/egoexo4d_ee4d_motion"

    ds_name = "ee4d"
    split = "val"
    ds = EE4D_Motion_Dataset(
        data_dir=data_dir,
        split=split,
        repre_type="v4_beta",
        cond_img_feat=True,
        cond_traj=True,
        window=80,
        img_feat_type="dinov2",
        cond_betas=False,
    )

    # save path
    save_path = f"{data_dir}/uniegomotion/ee_val_gt_for_evaluation.pkl"
    assert not os.path.exists(save_path), f"File already exists: {save_path}"

    all_preds = {}

    for idx in tqdm(range(0, len(ds), 10)):
        batch = careful_collate_fn([ds[idx]])
        pred_mdata = ds.ret_to_full_sequence(batch)

        seq_name = batch["misc"]["seq_name"][0]
        start_idx = batch["misc"]["start_idx"][0] // 3
        k = f"{seq_name}_start_{start_idx}"
        assert k not in all_preds
        all_preds[k] = {
            "smpl_params": pred_mdata["smpl_params_full"][0],
            "aria_traj_T": pred_mdata["aria_traj_T"][0],
        }

    # dump metric data
    joblib.dump(all_preds, save_path)


if __name__ == "__main__":
    save_params()
    pass
