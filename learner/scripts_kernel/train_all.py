import os 
from itertools import product
from tqdm import tqdm

SAVE_DIR = "trained_kernel_models"
LOG_DIR = "logs_training_kernels"

os.makedirs(LOG_DIR, exist_ok=True)

configs = product(
    ["wl"],  # wl algorithms
    [1,2,3,4,5],  # iterations
    ["ig"],  # representation
    ["ferry", "blocksworld", "childsnack", "floortile", "miconic", "rovers", "satellite", "sokoban", "spanner", "transport"],  # domains
    ["linear-svr", "lasso", "ridge", "rbf-svr", "quadratic-svr", "cubic-svr", "mlp"],  # models
)

# deadend_configs = [("H", ""), ("D", "--deadends")]
deadend_configs = [("D", "--deadends")]

for wl, iterations, rep, domain, model in tqdm(configs):
    for target, flag in deadend_configs:
        desc = f"{wl}_{iterations}_{rep}_{model}_{target}_{domain}"
        save_file = f"{SAVE_DIR}/{desc}.joblib"
        log_file = f"{LOG_DIR}/{desc}.log"
        if os.path.exists(save_file) and os.path.exists(log_file):
            continue
        cmd = f"python3 train_kernel.py {domain} {flag} -k {wl} -l {iterations} -r {rep} -m {model} --save-file {save_file}"
        cmd = f"{cmd} > {log_file}"
        print(cmd)
        os.system(cmd)
