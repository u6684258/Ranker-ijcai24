import os
import argparse
from itertools import product

"""
2 SU per CPU/4GB per hour
+~> 30 minute + 4GB job = 1 SU
 ~> 30 minute + 8GB job = 2 SU
+~> 30 minute + 16GB job = 4 SU
+960 SU for 16GB and 240 configs
"""

_TIMEOUT = "00:10:00"
_TIMEOUT = "00:30:00"

_CONFIGS = product(
    ["1wl", "2gwl", "2lwl"],  # wl algorithms
    [1, 2, 3, 4, 5, 6, 7, 8],  # iterations
    [0],  # prunes
    ["ilg"],  # representation  
    [  # domains
        "ferry",
        "blocksworld",
        "childsnack",
        "floortile",
        "miconic",
        "rovers",
        "satellite",
        "sokoban",
        "spanner",
        "transport",
    ],
    [  # ml models
        "linear-svr", 
        # "quadratic-svr", 
        # "cubic-svr", 
        # "rbf-svr", 
        # "lasso", 
        # "ridge"
    ],
)

_LOG_DIR = "icaps24_train_logs"
_MODEL_DIR = "icaps24_wl_models"
_LOCK_DIR = "lock"

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

    for wl, iterations, prune, rep, domain, model in _CONFIGS:
        desc = "_".join([domain, rep, wl, str(iterations), str(prune), model, "H"])
        lock_file = f"{_LOCK_DIR}/{desc}.lock"
        log_file = f"{_LOG_DIR}/{desc}.log"
        model_save_file = f"{_MODEL_DIR}/{desc}.joblib"

        if (os.path.exists(model_save_file) and os.path.exists(log_file)) or (os.path.exists(lock_file)):
            skipped += 1
            continue

        if submitted >= e:
            to_go += 1
            continue

        with open(lock_file, "w") as f:
            pass

        cmd = f"python3 train_kernel.py -m {model} -r {rep} -d {domain} -k {wl} -l {iterations} -p {prune} --model-save-file {model_save_file}"

        cmd = (
            f"qsub -o {log_file} -j oe -l walltime={_TIMEOUT} -v "
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
