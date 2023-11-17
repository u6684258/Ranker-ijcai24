import os
from itertools import product
from tqdm import tqdm

SAVE_DIR = "kernel_training_data"
LOG_DIR = "logs_kernel_training_data"

os.makedirs(LOG_DIR, exist_ok=True)

configs = product(
    ["wl", "2wl"],  # wl algorithms
    [1, 2, 4, 8, 16],  # iterations
    [1, 2, 4, 8, 16],  # count prunes
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

# deadend_configs = [("H", ""), ("D", "--deadends")]
deadend_configs = [("H", "")]
# deadend_configs = [("D", "--deadends")]

for wl, iterations, prune, rep, domain in tqdm(list(configs)):
    for target, flag in deadend_configs:
        desc = "_".join([wl, str(iterations), str(prune), rep, domain, target])
        save_file = f"{SAVE_DIR}/{desc}.pkl"
        log_file = f"{LOG_DIR}/{desc}.log"
        if os.path.exists(save_file) and os.path.exists(log_file):
            continue
        cmd = f"python3 train_kernel.py {flag} -d {domain} -k {wl} -l {iterations} -r {rep} -p {prune} --data-save-file {save_file}"
        cmd = f"{cmd} > {log_file}"
        print(cmd)
        os.system(cmd)
