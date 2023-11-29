import os
import argparse
from itertools import product

"""
2 SU per CPU/4GB per hour
~> 30 minute + 8GB job = 2 SU
"""

_PBS_TIMEOUT = "00:31:30"

# 900 all problems / 300 per difficulty
# 1.8 KSU all problems / 0.6 KSU per difficulty
_DOMAINS = [
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
_DIFFICULTIES = [
    "easy",
    "medium",
    "hard",
]

def skip(domain, difficulty):
    if domain == "floortile":
        return difficulty in {"medium", "hard"}
    
    if domain != "ferry":
        return difficulty in {"hard"}
    
    return False

_LEARNING_MODELS = [
    "linear-svr",
    # "gp",
]
_REPRESENTATIONS = ["ilg"]
_WLS = [
    "1wl", 
    # "2gwl", 
    # "2lwl",
]
_ITERATIONS = [
    # 1, 
    4,
]

# inaccurate fd timer, rely on pbs script and postprocessing for timing out
_TIMEOUT = 360000  

_LOG_DIR = "icaps24_test_logs"
_LOCK_DIR = "lock"
_MODEL_DIR = "icaps24_wl_models"
_GP_MODEL_DIR = "icaps24_gp_models"

os.makedirs(_LOG_DIR, exist_ok=True)
os.makedirs(_LOCK_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", type=int, default=1)
    args = parser.parse_args()
    to_go = 0
    skipped = 0
    submitted = 0
    model_missing = 0

    e = args.e

    CONFIGS = list(
        product(
            _LEARNING_MODELS,
            _REPRESENTATIONS,
            _WLS,
            _ITERATIONS,
            _DOMAINS,
            _DIFFICULTIES,
        )
    )

    missing_models = set()

    for config in CONFIGS:
        learning_model, rep, wl, iterations, domain, difficulty = config

        if skip(domain, difficulty):
            continue
        
        if learning_model == "gp":
            mf = f"{_GP_MODEL_DIR}/{domain}_gp.pkl"
        else:
            mf = f"{_MODEL_DIR}/{domain}_{rep}_{wl}_{iterations}_0_{learning_model}_H.pkl"

        df = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/domain.pddl"
        problem_dir = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/testing/{difficulty}"

        for file in sorted(os.listdir(problem_dir)):
        
            if not os.path.exists(mf):
                model_missing += 1
                missing_models.add(mf)
                continue

            pf = f"{problem_dir}/{file}"

            problem = os.path.basename(pf).replace(".pddl", "")

            # check whether to skip
            desc = f'{domain}_{difficulty}_{problem}_{rep}_{wl}_{iterations}_{learning_model}_H'
            log_file = f"{_LOG_DIR}/{desc}.log"
            lock_file = f"{_LOCK_DIR}/{desc}.lock"

            if os.path.exists(log_file) or os.path.exists(lock_file):
                skipped += 1
                continue

            if submitted >= e:
                to_go += 1
                continue

            # submit
            with open(lock_file, "w") as f:
                pass

            cmd = f"python3 run_kernel.py {df} {pf} {mf}  --timeout {_TIMEOUT}"

            cmd = (
                f"qsub -o {log_file} -j oe -l walltime={_PBS_TIMEOUT} -v "
                + f'CMD="{cmd}",'
                + f'LOCK_FILE="{lock_file}" '
                + f"pbs_scripts/job.sh"
            )
            os.system(cmd)
            print(log_file)
            submitted += 1

    if len(missing_models) > 0:
        print("missing models:")
        for m in sorted(missing_models):
            print(m)
            
    print("submitted:", submitted)
    print("skipped:", skipped)
    print("to_go:", to_go)
    print("model_missing:", model_missing)

if __name__ == "__main__":
    main()
