#!/bin/bash
 
#PBS -P xb83
#PBS -q express
#PBS -l ncpus=1
#PBS -l mem=8GB
#PBS -l jobfs=20GB
#PBS -l wd
#PBS -M dongbang4204@gmail.com

rm icaps24_test_logs.zip
zip -r icaps24_test_logs.zip icaps24_test_logs

# rm $LOCK_FILE
