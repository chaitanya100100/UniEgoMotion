import sys
import copy
import subprocess
import os


def run_cmd(cmd):
    print(" ".join(cmd))
    os.system(" ".join(cmd))
    # result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    # print(f"STDOUT {result.stdout}")
    # print(f"STDERR {result.stderr}")


def main():

    old_argv = copy.deepcopy(sys.argv)

    from config.defaults import get_cfg

    cfg = get_cfg()
    assert cfg.TRAIN.EXP_PATH is not None
    assert cfg.MODEL.CKPT_PATH is not None or cfg.MODEL.TRAJ_CKPT_PATH is not None

    for task in ["recon", "gen", "fore"]:
        # for task in ["recon"]:

        cmd = ["python", "eval/save_uem_preds.py"] + old_argv[1:] + ["TRAIN.EVAL_TASK", task]
        run_cmd(cmd)

        cmd = ["python", "eval/compute_3d_metrics.py", "--EXP_PATH", cfg.TRAIN.EXP_PATH, "--EVAL_TASK", task]
        if cfg.TRAIN.EVAL_SUFFIX is not None and cfg.TRAIN.EVAL_SUFFIX != "":
            cmd += ["--EVAL_SUFFIX", cfg.TRAIN.EVAL_SUFFIX]
        run_cmd(cmd)

        cmd = ["python", "eval/compute_semantic_metrics.py", "--EXP_PATH", cfg.TRAIN.EXP_PATH, "--EVAL_TASK", task]
        if cfg.TRAIN.EVAL_SUFFIX is not None and cfg.TRAIN.EVAL_SUFFIX != "":
            cmd += ["--EVAL_SUFFIX", cfg.TRAIN.EVAL_SUFFIX]
        run_cmd(cmd)


if __name__ == "__main__":
    main()
