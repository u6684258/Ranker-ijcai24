import os 
import time
from itertools import product

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

SLURM_SCRIPT="slurm/cluster1_job_stromboli_cpu"

LOG_DIR = "icaps24_train_logs"
MODEL_DIR = "icaps24_wl_models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

WLS = ["1wl", "2lwl", "2gwl"]


_CONFIGS = [
    # ("2lwl", 1, 0, "ilg", "floortile", "linear-svr"),
    ("1wl", 4, 0, "ilg", "sokoban", "gp")
]

# for domain, wl in product(DOMAINS, WLS):
#     _CONFIGS.append(
#         (wl, 4, 0, "ilg", domain, "linear-svr")
#     )

for wl, iterations, prune, rep, domain, model in _CONFIGS:
    desc = "_".join([domain, rep, wl, str(iterations), str(prune), model, "H"])
    model_save_file = f"{MODEL_DIR}/{desc}.joblib"
    log_file = f"{LOG_DIR}/{desc}.log"
    if model == "gp":
        train_script = "train_bayes.py"
    else:
        train_script = "train_kernel.py"
    submit_cmd = f'sbatch --job-name=tr_{desc} --output={log_file} {SLURM_SCRIPT} "python3 {train_script} -m {model} -r {rep} -d {domain} -k {wl} -l {iterations} -p {prune} --model-save-file {model_save_file}"'
    os.system(submit_cmd)
    print(f"submit {desc}")
