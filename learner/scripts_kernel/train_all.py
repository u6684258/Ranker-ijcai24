import os
import sys
from itertools import product
from tqdm import tqdm

DOMAINS = [  # domains
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
]

# save_dir = "icaps24_gp_models"
# log_dir = "icaps24_gp_train_logs"
# os.makedirs(save_dir, exist_ok=True)
# os.makedirs(log_dir, exist_ok=True)

# for domain in DOMAINS:
#     os.system(f"python3 train_bayes.py -d {domain} -m gp --model-save-file {save_dir}/{domain}_gp.pkl | tee {log_dir}/{domain}_gp.log")
# assert 0

SAVE_DIR = "icaps24_wl_models"
LOG_DIR = "icaps24_train_logs"

os.makedirs(LOG_DIR, exist_ok=True)

configs = [
    # ("1wl", 4, 0, "ilg", "quadratic-svr", "none"),
    # ("1wl", 4, 0, "ilg", "cubic-svr", "none"),
    # ("1wl", 4, 0, "ilg", "rbf-svr", "none"),
    # ("1wl", 4, 0, "ilg", "gp", "none"),
    ("3lwl", 4, 0, "ilg", "linear-svr", "none"),
    # ("1wl", 4, 0, "ilg", "mip", "schema"),
    # ("1wl", 4, 0, "ilg", "linear-svr", "none"),
    # ("1wl", 4, 0, "ilg", "blr", "none"),
]

# deadend_configs = [("H", "?"), ("D", "--deadends")]
# deadend_configs = [("D", "--deadends")]
deadend_configs = [("H", "")]

for domain in DOMAINS:
    for k, l, p, r, m, schema_strategy in tqdm(list(configs)):
        for t, flag in deadend_configs:
            desc = f"{domain}_{r}_{k}_{l}_{p}_{m}_{schema_strategy}_{t}"
            save_file = f"{SAVE_DIR}/{desc}.pkl"
            log_file = f"{LOG_DIR}/{desc}.log"
            # if os.path.exists(save_file) and os.path.exists(log_file):
            #     continue
            cmd = f"python3 train_kernel.py -d {domain} -k {k} -l {l} -r {r} -m {m} {flag} --schema {schema_strategy} --model-save-file {save_file}"
            cmd = f"{cmd} > {log_file}"
            print(cmd)

            try:
                os.system(cmd)
            except KeyboardInterrupt:
                print('Interrupted')
                try:
                    sys.exit(-1)
                except SystemExit:
                    os._exit(-1)
