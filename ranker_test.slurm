#! /bin/bash
#SBATCH --job-name=rank_all
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=u6684258@anu.edu.au     # Where to send mail
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
# #SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=240:00:00
#SBATCH --partition=planopt
#SBATCH --qos=planopt
#SBATCH --output=rnk_test_%j.log
# Standard output and error log
# blocksworld childsnack ferry floortile miconic rovers satellite sokoban spanner transport
pwd; hostname; date
echo "--- running ---"
for MODEL in gnn-rank
    do
    for DOMAIN in ferry
        do
            nohup singularity run -B ~/Codes/goose/learner/:/learner  \
                -B ~/Codes/goose/planners/FD-Hypernet-master/src/search/heuristics/state_evaluation.py:/planners/FD-Hypernet-master/src/search/heuristics/state_evaluation.py  \
                -B ~/Codes/goose/benchmarks/:/benchmarks \
                -B ~/Codes/goose/data/:/data \
                -B ~/Codes/goose/logs/:/logs \
                ~/Codes/goose/rank-gnn.sif \
                mean 4 ilg "$MODEL" "$DOMAIN" 0 1> ~/rank_logs_std/pbs-"$DOMAIN"-"$MODEL"-ilg_info.log 2> ~/rank_logs_err/pbs-"$DOMAIN"-"$MODEL"-ilg_err.log
        done
done
echo "--- done ---"
date
