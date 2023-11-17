import os
import argparse
from itertools import product

"""
2 SU per CPU/4GB per hour
~> 30 minute + 8GB job = 2 SU
"""

_CONFIGS = product(
    ["wl"],  # wl algorithms
    [1,2,3,4,6,8,12,16,24,32],  # iterations
    ["ig"],  # representation
    ["ferry", "blocksworld", "childsnack", "floortile", "miconic", "rovers", "satellite", "sokoban", "spanner", "transport"],  # domains
    ["linear-svr", "lasso", "ridge", "rbf-svr", "quadratic-svr", "cubic-svr", "mlp"],  # models
)

_LOG_DIR = "logs_training_kernels"
_LOCK_DIR = "lock"
_MODEL_DIR = "trained_kernel_models"

os.makedirs(_LOG_DIR, exist_ok=True)
os.makedirs(_LOCK_DIR, exist_ok=True)
os.makedirs(_MODEL_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", type=int, default=1)
    args = parser.parse_args()
    to_go = 0
    skipped = 0
    submitted = 0

    e = args.e

    for wl, iterations, rep, domain, model in _CONFIGS:
        desc = f"{wl}_{iterations}_{rep}_{model}_H_{domain}"
        save_file = f"{_MODEL_DIR}/{desc}.joblib"
        lock_file = f"{_LOCK_DIR}/{desc}.lock"
        log_file = f"{_LOG_DIR}/{desc}.log"

        if (os.path.exists(save_file) and os.path.exists(log_file)) or (os.path.exists(lock_file)):
            skipped += 1
            continue

        if submitted >= e:
            to_go += 1
            return

        with open(lock_file, "w") as f:
            pass

        cmd = f"python3 train_kernel.py {domain} -k {wl} -l {iterations} -r {rep} -m {model} --save-file {save_file}"

        cmd = (
            f"qsub -o {log_file} -j oe -v "
            + f'CMD="{cmd}",'
            + f'LOCK_FILE="{lock_file}" '
            + f"pbs_scripts/train_kernel_job.sh"
        )
        os.system(cmd)
        submitted += 1

    print("submitted:", submitted)
    print("skipped:", skipped)
    print("to_go:", to_go)

if __name__ == "__main__":
    main()
