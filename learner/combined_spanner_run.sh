cd ../planners/downward && python3 build.py
cd ../../learner

python3 run_kernel.py ../benchmarks/ipc2023-learning-benchmarks/spanner/domain.pddl ../benchmarks/ipc2023-learning-benchmarks/spanner/testing/medium/p10.pddl spanner_combined.pkl -s mq
