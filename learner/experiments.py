import os
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--rep', type=str, default="slg")
parser.add_argument('--domains', nargs='+', type=str, default=["blocks", "ferry", "gripper", "sokoban"])
args = parser.parse_args()
exp_root = os.path.dirname(os.path.realpath(__file__))
log_root = os.path.join(exp_root, "logs/train")
log_sub_dir = os.path.join(log_root, datetime.now().isoformat())
Path(log_sub_dir).mkdir(parents=True, exist_ok=True)
# parameters
model = args.rep

print(f"Selected domains: {args.domains}")

for i in range(3, 11):
    for domain in args.domains:
        cmd = f'python3 {exp_root}/train.py ' \
              f'-m RGNNBATRANK ' \
              f'-r {model} ' \
              f'-d rank-{domain} ' \
              f'-L {i} ' \
              f'--save-file rank-{domain} ' \
              f'--method batched_ranker ' \
              f'--fast-train'

        f = open(f"{log_sub_dir}/train_rank_{domain}_L{i}.logs", "w")

        print(f"Experiment log: {f}")

        subprocess.call(cmd.split(" "), stdout=f)

        cmd = f'python3 {exp_root}/train.py ' \
              f'-m RGNNBATRANK ' \
              f'-r {model} ' \
              f'-d rndrank-{domain} ' \
              f'-L {i} ' \
              f'--save-file rank-{domain} ' \
              f'--method ranker_random ' \
              f'--fast-train'

        f = open(f"{log_sub_dir}/train_rndrank_{domain}_L{i}.logs", "w")

        print(f"Experiment log: {f}")

        subprocess.call(cmd.split(" "), stdout=f)

    # cmd = f'python3 {exp_root}/train.py -m RGNN -r {model} -d goose-{domain} --save' \
    #       f'-file goose-{domain} --fast-train'
    #
    # f = open(f"{log_sub_dir}/train_goose_{domain}.logs", "w")
    #
    # print(f"Experiment log: {f}")
    #
    # subprocess.call(cmd.split(" "), stdout=f)

