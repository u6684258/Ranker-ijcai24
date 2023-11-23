import os
from itertools import product
from tqdm import tqdm

SAVE_DIR = "icaps24_wl_models"
LOG_DIR = "icaps24_train_logs"

os.makedirs(LOG_DIR, exist_ok=True)

configs = product(
    ["1wl", "2gwl", "2lwl"],  # wl algorithms
    [1],  # iterations
    [0],  # prunes
    ["ilg"],  # representation
    [  # domains
        "blocksworld",
        "childsnack",
        "ferry",
        "floortile",
        "miconic",
        "rovers",
        "satellite",
        "sokoban",
        "spanner",
        "transport",
    ],
    [  # models
        "linear-svr",
        # "lasso",
        # "ridge",
        # "rbf-svr",
        # "quadratic-svr",
        # "cubic-svr",
        # "mlp",
    ],
)

# deadend_configs = [("H", "?"), ("D", "--deadends")]
# deadend_configs = [("D", "--deadends")]
deadend_configs = [("H", "")]

for k, l, p, r, domain, m in tqdm(list(configs)):
    for t, flag in deadend_configs:
        desc = f"{domain}_{r}_{k}_{l}_{p}_{m}_{t}"
        save_file = f"{SAVE_DIR}/{desc}.joblib"
        log_file = f"{LOG_DIR}/{desc}.log"
        if os.path.exists(save_file) and os.path.exists(log_file):
            continue
        cmd = f"python3 train_kernel.py -d {domain} -k {k} -l {l} -r {r} -m {m} {flag} --model-save-file {save_file}"
        cmd = f"{cmd} > {log_file}"
        print(cmd)
        os.system(cmd)
