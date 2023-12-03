cd ../planners/downward && python3 build.py
cd ../../learner

domain=$1
# mq_arg=$2  # mq or mqp
mq_arg=mq

python3 run_kernel.py ../benchmarks/ipc2023-learning-benchmarks/${domain}/domain.pddl ../benchmarks/ipc2023-learning-benchmarks/${domain}/testing/medium/p10.pddl icaps24_wl_models/${domain}_combined_gp.pkl -s $mq_arg
