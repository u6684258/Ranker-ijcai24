import multiprocessing
import os
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--rep', type=str, default="slg")
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--domains', nargs='+', type=str, default=["blocks", "ferry", "gripper", "sokoban"])
args = parser.parse_args()
exp_root = os.path.dirname(os.path.realpath(__file__))
log_root = os.path.join(exp_root, "logs/train")
log_sub_dir = os.path.join(log_root, datetime.now().isoformat())
Path(log_sub_dir).mkdir(parents=True, exist_ok=True)
# parameters
model = args.rep

print(f"Selected domains: {args.domains}")


def task(domain, layers):
    cmd = f'python3 {exp_root}/train.py ' \
          f'-m RGNNBATRANK ' \
          f'-r {model} ' \
          f'-d goose-{domain} ' \
          f'-L {layers} ' \
          f'--save-file rank-{domain} ' \
          f'--method batched_ranker ' \
          f'--fast-train'

    f = open(f"{log_sub_dir}/train_rank_{domain}_L{layers}.logs", "w")
    print(f"Experiment log: {f}")
    subprocess.call(cmd.split(" "), stdout=f)

    cmd = f'python3 {exp_root}/train.py -m RGNN -r {model} -d goose-{domain} ' \
          f'--save-file goose-{domain} -L {layers} --fast-train --method goose'

    f = open(f"{log_sub_dir}/train_goose_{domain}_L{layers}.logs", "w")
    print(f"Experiment log: {f}")
    subprocess.call(cmd.split(" "), stdout=f)


jobs = []
for domain in args.domains:
    for layer in range(args.layers, args.layers + 5):
        jobs.append((domain, layer))

count = 4
pool = multiprocessing.Pool(processes=count)
matrices = []
r = pool.starmap_async(task, jobs, error_callback=lambda x: print(x))
r.wait()
