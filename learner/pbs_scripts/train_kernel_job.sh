#!/bin/bash
 
#PBS -P xb83
#PBS -q normal
#PBS -l walltime=00:33:00
#PBS -l ncpus=4
#PBS -l mem=16GB
#PBS -l jobfs=20GB
#PBS -l wd
#PBS -M dongbang4204@gmail.com

module load python3/3.10.4
source /scratch/xb83/dc6693/venv_goose/bin/activate

python3 scripts_kernel/train_all.py
