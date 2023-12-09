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

SVR = "ilg2_1wl_4_0_linear-svr_none_H"
GPR = "ilg2_1wl_4_0_gp_none_H"
GPR_STD = "ilg2_1wl_4_0_gp_none_H_std"

RBF = "ilg2_1wl_4_0_rbf-svr_none_H"
LWL = "ilg2_2lwl_4_0_linear-svr_none_H"

MIP = "ilg2_1wl_4_0_mip_schema_H"
MQ_SVR = "ilg2_combined_svr"
MQ_GPR = "ilg2_combined_gpr"

# MIP = "ilg_1wl_4_0_mip_schema_H"
# SVR = "ilg_1wl_4_0_linear-svr_H"
# GPR = "ilg_1wl_4_0_gp_none_H"
# MQ_SVR = "combined"
# MQ_GPR = "combined_gp"

_MODELS = [
    # SVR,
    # GPR,

    GPR_STD,

    # RBF,
    # LWL,

    # MIP,
    # MQ_GPR,
    # MQ_SVR,
]

def skip(domain, difficulty, model):
    # skip BW and Sokoban for MIP models
    if domain in {"blocksworld", "sokoban"} and model in {MIP, MQ_GPR, MQ_SVR}:
        return True

    # skip Floortile non-easy
    if domain == "floortile":
        return difficulty in {"medium", "hard"}
    
    if domain not in {"ferry", "miconic", "blocksworld", "spanner"}:
        return difficulty in {"hard"}
     
    return False

_LOG_DIR = "icaps24_test_logs"
_LOCK_DIR = "lock"
_MODEL_DIR = "icaps24_wl_models"

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
            _MODELS,
            _DOMAINS,
            _DIFFICULTIES,
        )
    )

    missing_models = set()

    for config in CONFIGS:
        model, domain, difficulty = config

        if skip(domain, difficulty, model):
            continue
        
        mf = f"{_MODEL_DIR}/{domain}_{model}.pkl"
        mf = mf.replace("_std", "")

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
            desc = f'{domain}_{difficulty}_{problem}_{model.replace("_0_", "_")}'
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

            cmd = f"python3 run_kernel.py {df} {pf} {mf} "

            if model in {MQ_SVR, MQ_GPR}:
                cmd += "-s mq "
            if model == GPR_STD:
                cmd += "--std "

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
