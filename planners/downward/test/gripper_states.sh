DOMAIN="../../../benchmarks/ipc23/blocksworld/domain.pddl"
INSTANCE="../../../benchmarks/ipc23/blocksworld/training/easy/p01.pddl"
SOLUTION="../../../benchmarks/ipc23/solutions/blocksworld/training/easy/p01.plan"
CONFIG="1"

# 0: slg, 1: flg, 2: llg, 3: glg

export GOOSE="$HOME/code/goose/learner"

export PLAN_INPUT_PATH=$SOLUTION
export STATES_OUTPUT_PATH=./plan_siblings.state
./../fast-downward.py --sas-file plan.sas $DOMAIN $INSTANCE --search 'perfect_with_siblings([blind()])'
export STATES_OUTPUT_PATH=./plan.state
./../fast-downward.py --sas-file plan.sas $DOMAIN $INSTANCE --search 'perfect([blind()])'