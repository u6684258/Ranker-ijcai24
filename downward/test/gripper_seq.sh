DOMAIN="../../benchmarks/goose/gripper/domain.pddl"
INSTANCE="../../benchmarks/goose/gripper/val/gripper-n11.pddl"
CONFIG="1"

# 0: slg, 1: flg, 2: llg, 3: glg

export GOOSE="$HOME/PycharmProjects/goose/learner"

cd test


./../fast-downward.py --search-time-limit 600 \
  --sas-file test.sas_file \
  --plan-file test.plan \
  /home/ming/PycharmProjects/goose/learner/../benchmarks/goose/ferry/domain.pddl \
  /home/ming/PycharmProjects/goose/learner/../benchmarks/goose/ferry/test/p-l85-c85-s4.pddl \
  --search 'batch_eager_greedy([goose(model_path="/home/ming/PycharmProjects/goose/learner/trained_models_gnn/test-ferry.dt",
  model_type="gnn", domain_file="/home/ming/PycharmProjects/goose/learner/../benchmarks/goose/ferry/domain.pddl",
  instance_file="/home/ming/PycharmProjects/goose/learner/../benchmarks/goose/ferry/test/p-l85-c85-s4.pddl")])'
