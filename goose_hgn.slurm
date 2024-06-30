#! /bin/bash
#SBATCH --job-name=rk_ipc_hgn_t
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=u6684258@anu.edu.au     # Where to send mail
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# #SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=240:00:00
#SBATCH --partition=planopt
#SBATCH --qos=planopt
#SBATCH --output=rk_ipc_%j.log
# Standard output and error log
pwd; hostname; date
echo "--- running ---"
# DOMAIN=blocksworld
for DOMAIN in blocksworld childsnack ferry floortile miconic \
        rovers satellite sokoban spanner transport
# for DOMAIN in rovers sokoban transport
do
        singularity run -B ~/Codes/goose/learner/:/learner  \
                -B ~/Codes/goose/planners/FD-Hypernet-master/src/search/heuristics/state_evaluation.py:/planners/FD-Hypernet-master/src/search/heuristics/state_evaluation.py  \
                -B ~/Codes/goose/benchmarks/:/benchmarks \
                -B ~/Codes/goose/data/:/data \
                -B ~/Codes/goose/logs/:/logs \
                ~/Codes/goose/goose_hgn.sif \
                mean 4 llg hgn-loss "$DOMAIN" 0
done
echo "--- done ---"
date
