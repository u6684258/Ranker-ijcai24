import multiprocessing
import os
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--rep', type=str, default="slg")
parser.add_argument('--method', type=str, default="goose")
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--domains', nargs='+', type=str, default=["blocks", "ferry", "gripper", "sokoban"])
args = parser.parse_args()
exp_root = os.path.dirname(os.path.realpath(__file__))
log_root = os.path.join(exp_root, f"logs/{datetime.now().isoformat()}")
log_sub_dir = os.path.join(log_root, f"train")
Path(log_sub_dir).mkdir(parents=True, exist_ok=True)
# parameters
model = args.rep

print(f"Selected domains: {args.domains}")


def task(domain, layers, method):
    if method == "ranker":
        cmd = f'python3 {exp_root}/train.py ' \
              f'-m RGNNBATCOORDRANK ' \
              f'-r {model} ' \
              f'-d goose-{domain} ' \
              f'-L {layers} ' \
              f'--log-root {log_root} ' \
              f'--save-file rank-{domain}-L{layers}-coord ' \
              f'--method batched_coord_ranker ' \
              f'--fast-train'
              # f'--test-only'


        f = open(f"{log_sub_dir}/train_rank_{domain}_L{layers}.logs", "w")
        print(f"Experiment log: {f}")
        subprocess.call(cmd.split(" "), stdout=f)
    elif method == "goose":
        cmd = f'python3 {exp_root}/train.py -m RGNN -r {model} -d goose-{domain} ' \
              f'--save-file goose-{domain}-L{layers} -L {layers} --log-root {log_root} ' \
              f'--fast-train ' \
              f'--method goose'

        f = open(f"{log_sub_dir}/train_goose_{domain}_L{layers}.logs", "w")
        print(f"Experiment log: {f}")
        subprocess.call(cmd.split(" "), stdout=f)


jobs = []
for domain in args.domains:
    for layer in range(args.layers, args.layers + 7):
        jobs.append((domain, layer))

count = 1
pool = multiprocessing.Pool(processes=count)
matrices = []
r = pool.starmap_async(task, jobs, error_callback=lambda x: print(x))
r.wait()

