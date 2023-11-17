import os
import argparse
from itertools import product

"""
2 SU per CPU/4GB per hour
~> 30 minute + 8GB job = 2 SU
"""

_TIMEOUT = "00:30:00"

_CONFIGS = product(
    ["wl", "2wl"],  # wl algorithms
    [1, 2, 4, 8, 16],  # iterations
    [0, 1, 2, 4, 8, 16],  # count prunes
    ["ig"],  # representation
    [
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
    ],  # domains
)

_DATA_DIR = "kernel_training_data"
_LOG_DIR = "logs_kernel_training_data"
_LOCK_DIR = "lock"

os.makedirs(_LOG_DIR, exist_ok=True)
os.makedirs(_LOCK_DIR, exist_ok=True)
os.makedirs(_DATA_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", type=int, default=1)
    args = parser.parse_args()
    to_go = 0
    skipped = 0
    submitted = 0

    e = args.e

    for wl, iterations, prune, rep, domain in _CONFIGS:
        desc = "_".join([wl, str(iterations), str(prune), rep, domain, "H"])
        save_file = f"{_DATA_DIR}/{desc}.pkl"
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

        cmd = f"python3 train_kernel.py -d {domain} -k {wl} -l {iterations} -r {rep} -p {prune} --data-save-file {save_file}"

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
