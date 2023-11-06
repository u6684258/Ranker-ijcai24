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

    # if method == "ranker":
    # cmd = f'python3 {exp_root}/train.py ' \
    #       f'-m RGNNBATCOORDRANK ' \
    #       f'-r llg ' \
    #       f'-d goose-{domain} ' \
    #       f'-L {layers} ' \
    #       f'--log-root {log_root} ' \
    #       f'--save-file rank-{domain}-L{layers}-coord ' \
    #       f'--method batched_coord_ranker ' \
    #       f'--fast-train'
    #       # f'--test-only'
    #
    #
    # f = open(f"{log_sub_dir}/train_rank_llg_{domain}_L{layers}.logs", "w")
    # print(f"Experiment log: {f}")
    # subprocess.call(cmd.split(" "), stdout=f)

    cmd = f'python3 {exp_root}/train.py ' \
          f'-m RGNNBATCOORDRANK ' \
          f'-r slg ' \
          f'-d goose-{domain} ' \
          f'-L {layers} ' \
          f'--log-root {log_root} ' \
          f'--save-file rank-{domain}-L{layers}-coord ' \
          f'--method batched_coord_ranker ' \
          f'--fast-train'
    # f'--test-only'

    f = open(f"{log_sub_dir}/train_rank_slg_{domain}_L{layers}.logs", "w")
    print(f"Experiment log: {f}")
    subprocess.call(cmd.split(" "), stdout=f)

    # elif method == "goose":
    cmd = f'python3 {exp_root}/train.py -m RGNN -r llg -d goose-{domain} ' \
          f'--save-file goose-llg-{domain}-L{layers} -L {layers} --log-root {log_root} ' \
          f'--fast-train ' \
          f'--method goose'

    f = open(f"{log_sub_dir}/train_goose_llg_{domain}_L{layers}.logs", "w")
    print(f"Experiment log: {f}")
    subprocess.call(cmd.split(" "), stdout=f)

    cmd = f'python3 {exp_root}/train.py -m RGNN -r slg -d goose-{domain} ' \
          f'--save-file goose-slg-{domain}-L{layers} -L {layers} --log-root {log_root} ' \
          f'--fast-train ' \
          f'--method goose'

    f = open(f"{log_sub_dir}/train_goose_slg_{domain}_L{layers}.logs", "w")
    print(f"Experiment log: {f}")
    subprocess.call(cmd.split(" "), stdout=f)

    # elif method == "hgn":
    cmd = f'python3 {exp_root}/train_hgn.py -m HGNN -r hgn -d goose-{domain} ' \
          f'--save-file hgn-{domain}-L{layers} -L {layers} --log-root {log_root} ' \
          f'--fast-train ' \
          f'--method hgn'

    f = open(f"{log_sub_dir}/train_hgn_{domain}_L{layers}.logs", "w")
    print(f"Experiment log: {f}")
    subprocess.call(cmd.split(" "), stdout=f)

    # elif method == "hgn_ranker":
    cmd = f'python3 {exp_root}/train_hgn.py -m HGNNRANK -r hgn_ranker -d goose-{domain} ' \
          f'--save-file hgn-rank-{domain}-L{layers} -L {layers} --log-root {log_root} ' \
          f'--fast-train ' \
          f'--method hgn_ranker'

    f = open(f"{log_sub_dir}/train_hgn_rank_{domain}_L{layers}.logs", "w")
    print(f"Experiment log: {f}")
    subprocess.call(cmd.split(" "), stdout=f)


jobs = []
for domain in args.domains:
    for layer in [4, 7, 10, 13]:
        jobs.append((domain, layer, args.method))

count = 1
pool = multiprocessing.Pool(processes=count)
matrices = []
r = pool.starmap_async(task, jobs, error_callback=lambda x: print(x))
r.wait()

