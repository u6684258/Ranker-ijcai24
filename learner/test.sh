DOMAINS=("$@")

python3 experiments.py --domains "${DOMAINS[@]}" > "exp_log.txt"