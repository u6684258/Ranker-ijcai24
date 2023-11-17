#!/bin/bash
 
#PBS -P xb83
#PBS -q normal
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l jobfs=20GB
#PBS -l wd
#PBS -M dongbang4204@gmail.com

module load python3/3.10.4
source /scratch/xb83/dc6693/venv_goose/bin/activate

$CMD

rm $LOCK_FILE
