import os 
import time

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

REP = "ilg"
LAYERS = 4
AGGR = "mean"

rep = REP
layers = LAYERS
aggr = AGGR

SLURM_SCRIPT="slurm/cluster1_job_gpusrv5_a6000"

def main():
    while True:
        os.system("date")
        queue_status = list(os.popen('squeue -u u6942650 -o "%30j %5t %10M %R"').readlines())
        queued_running_or_finished_configs = set()

        for line in queue_status:
            print(line.replace("\n", ""))
            queued_running_or_finished_configs.add(line.split()[0])

        for f in os.listdir("icaps24_slurm"):
            queued_running_or_finished_configs.add(f.replace(".log", ""))

        jobs_total = len(queue_status) - 1

        submit = 0
        for domain in DOMAINS:
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
