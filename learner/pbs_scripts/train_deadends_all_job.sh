#!/bin/bash
 
#PBS -P xb83
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l jobfs=20GB
#PBS -l wd
#PBS -M dillon.chen@anu.edu.au

module load python3/3.10.4
source /scratch/xb83/dc6693/venv_goose/bin/activate
python3 scripts_kernel/train_all.py
