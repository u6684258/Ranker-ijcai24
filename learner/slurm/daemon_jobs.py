import os 
import time
from itertools import product

DOMAINS = [
    "blocksworld",
    "childsnack",
    "ferry",
    # "floortile",
    "miconic",
    "rovers",
    "satellite",
    "sokoban",
    "spanner",
    "transport",
]

REPS = ["ilg", "llg"]
# LAYERS = [1, 4]
LAYERS = [4]
AGGRS = ["mean", "max"]

SLURM_SCRIPT="slurm/cluster1_job_gpusrv5_a6000"

CONFIGS = list(product(DOMAINS, LAYERS, AGGRS, REPS))

def main():
    while True:
        os.system("date")
        queue_status = list(os.popen('squeue -u u6942650 -o "%10i %30j %5t %10M %R"').readlines())
        queued_running_or_finished_configs = set()
        jobs_total = 0

        for line in queue_status:
            print(line.replace("\n", ""))
            queued_running_or_finished_configs.add(line.split()[1])
            if "gpusrv-3" in line or "gpusrv-5" in line or "QOSMaxJobsPerUserLimit" in line:
                jobs_total += 1

        for f in os.listdir("icaps24_slurm"):
            queued_running_or_finished_configs.add(f.replace(".log", ""))

        submit = 0
        for domain, layers, aggr, rep in CONFIGS:
            desc = f"{rep}_{layers}_{aggr}_{domain}"
            if desc in queued_running_or_finished_configs:
                continue
            if submit >= 8 - jobs_total:
                continue
            log_file = f"icaps24_slurm/{desc}.log"
            submit_cmd = f'sbatch --job-name={desc} --output={log_file} {SLURM_SCRIPT} "python3 scripts_gnn/train_test_ipc2023.py -r {rep} -d {domain} -a {aggr} -l {layers}"'
            os.system(submit_cmd)
            print(f"submit {desc}")
            submit += 1

        time.sleep(30)
    pass

if __name__ == "__main__":
    main()
