module load python3/3.10.4
source /scratch/xb83/dc6693/venv_goose/bin/activate

module load gcc/11.1.0
module load cmake/3.21.4
module load boost/1.80.0

python3 build.py
