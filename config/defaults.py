from yacs.config import CfgNode as CN

_C = CN()

_C.DATA = CN()
_C.DATA.DATA_DIR = "/vision/u/chpatel/data/egoexo4d_ee4d_motion/"
_C.DATA.DATASET_NAME = None
_C.DATA.BATCH_SIZE = 32
_C.DATA.NUM_WORKERS = 4
_C.DATA.WINDOW = 80  # Length of the motion sequence. At 10 fps, window of 80 is 8 seconds.
_C.DATA.REPRE_TYPE = "v4_beta"  # motion representation type
_C.DATA.COND_IMG_FEAT = False  # whether to condition on image features
_C.DATA.COND_TRAJ = True  # whether to condition on aria trajectory
# whether to condition on betas. We do NOT condition on betas in the paper but rather predict them.
_C.DATA.COND_BETAS = False
_C.DATA.IMG_FEAT_TYPE = "dinov2"  # image feature type if conditioning on image features

_C.MODEL = CN()
_C.MODEL.CKPT_PATH = None
_C.MODEL.PREDICT_XSTART = True
_C.MODEL.DIFFUSION_STEPS = 1000
_C.MODEL.NOISE_SCHEDULE = "cosine"
_C.MODEL.MODEL_NAME = "uem"  # uem, lstm, unet
_C.MODEL.LEARN_TRAJ = False  # trajectory model for two-stage model
_C.MODEL.TRAJ_CKPT_PATH = None  # for two-stage model
_C.MODEL.MOTION_CKPT_PATH = None  # for two-stage model
_C.MODEL.ENCODER_TSFM = None  # use "add" for ablation with tsfm encoder
_C.MODEL.LSTM_TYPE = "gen"  # ["gen", "fore"]. If model is lstm, this specifies the task lstm is trained for.
_C.MODEL.FINETUNE_TYPE = None
_C.MODEL.ZERO_MASK_TOKEN = False  # whether to use zero mask token instead of a learnable one.

_C.TRAIN = CN()
_C.TRAIN.LR = 3.0e-5
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.USE_CKPT_LR = False  # whether to use lr from the checkpoint rather than the config lr.
_C.TRAIN.EXP_PATH = None  # experiment log path to save logs and checkpoints

_C.TRAIN.NUM_EPOCHS = 200
_C.TRAIN.LOG_EVERY_N_STEPS = 50
# _C.TRAIN.VAL_CHECK_INTERVAL = 1.0
_C.TRAIN.CHECK_VAL_EVERY_N_EPOCHS = 1
# _C.TRAIN.SAVE_EVERY_N_STEPS = None
_C.TRAIN.SAVE_EVERY_N_EPOCHS = 10
_C.TRAIN.ONLY_VALIDATE = False
_C.TRAIN.NUM_GPUS = 1

_C.TRAIN.EVAL_SUFFIX = ""  # suffix to append to the evaluation and visualization results file
_C.TRAIN.EVAL_TASK = None  # task to evaluate or visualize. Should be one of ["recon", "gen", "fore"]
_C.TRAIN.COND_SCALE = None  # classifier free guidance scale. We do not use this for UniEgoMotion evaluation.


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def get_cfg():
    import sys
    import warnings

    # Example command: python train/train_uem.py CONFIG ./config/uem.yaml TRAIN.EXP_PATH ./exp/uem_v4b_dinov2 TRAIN.LR 1e-4

    warnings.filterwarnings(
        "ignore", message=".*You are using `torch.load` with `weights_only=False`.*", category=FutureWarning
    )

    cfg = get_cfg_defaults()
    argv = sys.argv.copy()

    if len(argv) > 1:
        if argv[1] == "CONFIG":
            cfg.merge_from_file(argv[2])
            argv = argv[3:]
        else:
            argv = argv[1:]
        cfg.merge_from_list(argv)
    cfg.freeze()
    print(cfg.dump())
    return cfg
