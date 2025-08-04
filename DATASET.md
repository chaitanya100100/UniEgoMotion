# EE4D-Motion Dataset

The **EE4D-Motion** dataset provides SMPLX parameters for the EgoExo4D takes (each video in the EgoExo4D dataset is referred to as a *take*). It is designed to be used alongside other EgoExo4D annotations for various 3D motion tasks. In the **UniEgoMotion** project, we use it for egocentric motion reconstruction, forecasting, and generation. EE4D-Motion is curated through a comprehensive data processing pipeline that includes bounding box detection, 2D keypoint estimation, single-view HMR prediction, a two-stage SMPLX fitting, and several quality control measures.

We provide the dataset in the following two formats. Download as necessary.
- [ee4d_motion.zip (Link TBD)](): Raw EE4D-Motion annotations for total 4792 takes. This is not required for running or training UniEgoMotion but may be useful for your own research.
- [ee4d_motion_uniegomotion.zip (Link TBD)](): Processed and filtered EE4D-Motion data for UniEgoMotion running or training UniEgoMotion. The filtered data has 2032 training takes and 560 validation takes.

## Dataset Organization

From `ee4d_motion.zip`:
- `<data_root>/ee4d_motion/*` Take-wise EE4D-Motion annotations.

From `ee4d_motion_uniegomotion.zip`:
- `<data_root>/annotations/splits.json` Information about dataset splits. Same as the original EgoExo4D dataset.
- `<data_root>/takes.json` Metadata for each take. Same as the original EgoExo4D dataset.
- `<data_root>/uniegomotion/ee_<train/val>.pt` Filtered and processed EE4D-Motion data for UniEgoMotion training/validation.
- `<data_root>/uniegomotion/egoview_dinov2_<train/val>.pt` DINOv2 features for egocentric video frames in the training/validation set.
- `<data_root>/uniegomotion/v4_beta_ee_train_stats.pt` Normalization statistics for the UniEgoMotion model using a head-centric motion representation. See `utils/representation_utils.py` for more details.
- `<data_root>/uniegomotion/ee_val_gt_for_evaluation.pkl` Processed validation data for evaluation.

> **Note:** The files `annotations/splits.json` and `takes.json` are identical to those in the original EgoExo4D dataset. They are included here to allow you to run experiments and visualizations even if you do not have access to the full EgoExo4D dataset. If you do have the EgoExo4D dataset, it is recommended to copy the `ee4d_motion` and `uniegomotion` directories into your EgoExo4D dataset directory (i.e., set `<data_root>` to your EgoExo4D data directory).

The file `dataset/egoexo4d_take_dataset.py` provides several utility functions to load egocentric and exocentric video frames, undistort them, and access the calibrated extrinsics and intrinsics for each camera (including the ego camera). For example usage, see `run/vis_ee4d_motion_sample_overlay.py`.

## Visualize Raw EE4D-Motion Annotations on EgoExo4D Videos

> You should skip this step if you do not have EgoExo4D dataset videos.

Let `<data_root>` be the root directory of the EgoExo4D dataset, which contains the `ee4d_motion` folder. Run the following command to visualize the motion annotations. Visualize shorter takes, such as `cmu_bike02_4`, rather than longer ones. The script `run/vis_ee4d_motion_sample_overlay.py` also serves as an example of how to align EE4D-Motion annotations with other EgoExo4D assets.
```
python run/vis_ee4d_motion_sample_overlay.py \
--data_dir <data_root> \
--take_name cmu_bike02_4 \
--out_dir <vis_output_dir>
```

## Visualize Processed EE4D-Motion Annotations with Aria Trajectory

Open `run/vis_ee4d_motion_processed_sample_3d.py` and set the appropriate path to the processed data. Run the script as shown below to visualize a motion segment in 3D along with the Aria trajectory. This does not require the EgoExo4D dataset videos.
```
python run/vis_ee4d_motion_processed_sample_3d.py
```


## Dataset Processing Information

> **Note:** These steps have already been executed, and the output is included in the provided data.

- Raw EE4D-Motion annotations (in `ee4d_motion.zip`) may be inaccurate when the person moves out of view, is visible in less than 2 exocentric cameras, is far away (basketball), or if some body parts are not visible (covid test). Such segments are filtered out for UniEgoMotion training. We use `run/process_for_uniegomotion.py` to generate `uniegomotion/ee_<train/val>.pt` for UniEgoMotion training/validation. Note that we also exclude Rock Climbing takes as the motion is not on the floor.
    - During this processing, each take is split into several good segments, discarding those with poor optimization results. As a result, each sequence is named using the format `<take_name>___<start_frame_index>___<end_frame_index>`. Frame indices are according to 30fps and inclusive. But the EE4D-Motion dataset is provided at 10fps. For example, `uniandes_basketball_003_42___585___825` sequence will have smpl motion sequence of length `(825 - 585) / 3 + 1 = 81` at 10fps.
- The `dataset.ee4d_motion_dataset.save_statistics` function is used to compute data statistics for model input normalization. The results are saved as `uniegomotion/v4_beta_ee_train_stats.pt`.
- The `run.extract_img_feats.run_dinov2` function extracts DINOv2 features for each take. These features are then consolidated using the `run.extract_img_feats.merge_dinov2` function, producing a single feature file per split: `uniegomotion/egoview_dinov2_<train/val>.pt`.
- To keep evaluation time manageable, we use every 10th validation sample, which corresponds to evaluating an 8-second segment every 20 seconds. The script `eval/save_gt_for_eval.py` formats the evaluation data and saves it as `uniegomotion/ee_val_gt_for_evaluation.pkl`.
