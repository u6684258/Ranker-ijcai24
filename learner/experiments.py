import os
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--domains', nargs='+', type=str, default=["blocks", "ferry", "gripper", "sokoban"])
args = parser.parse_args()
exp_root = os.path.dirname(os.path.realpath(__file__))
log_root = os.path.join(exp_root, "logs")
Path(log_root).mkdir(parents=True, exist_ok=True)

print(f"Selected domains: {args.domains}")
for domain in args.domains:
    cmd = f'python3 {exp_root}/train.py -m RGNNRANK -r sdg-el -d goose-{domain}-only --domain-name {domain} --save-file test-{domain} --ranker True'

    f = open(f"{log_root}/exp_{domain}_{datetime.now().isoformat()}.logs", "w")

    subprocess.call(cmd.split(" "), stdout=f)
