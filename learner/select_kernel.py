import os
from itertools import product
from tqdm import tqdm
from util.scrape_log import scrape_kernel_train_log

_MODELS = "icaps24_wl_models"
_TRAIN_LOGS = "icaps24_train_logs"

ITERATIONS = [1, 2, 3, 4, 5, 6, 7, 8]
WL = ["1wl", "2gwl", "2lwl"]
DOMAINS = [
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
REP = "ilg"
PRUNE = 0
MODEL = "linear-svr"

IMPROVEMENT_REQUIREMENT = 0.1

# deadend_configs = [("H", "?"), ("D", "--deadends")]
# deadend_configs = [("D", "--deadends")]
# deadend_configs = [("H", "")]

for domain in DOMAINS:
    for wl in WL:
        best_val_loss = float("inf")
        best_iteration = 0

        for itr in ITERATIONS:
            log_file = f"{_TRAIN_LOGS}/{domain}_{REP}_{wl}_{itr}_{PRUNE}_{MODEL}_H.log"
            # if not os.path.exists(log_file):
            #     continue
            stats = scrape_kernel_train_log(log_file)
            if not "val_mse" in stats or not "train_mse" in stats:
                continue
            cur_val_loss = stats["val_mse"]
            cur_train_loss = stats["train_mse"]
            if cur_val_loss < best_val_loss * (1 - IMPROVEMENT_REQUIREMENT):
                best_val_loss = cur_val_loss
                best_iteration = itr

        print(domain, wl, best_iteration)
