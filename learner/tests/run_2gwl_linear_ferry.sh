MODEL=linear-svr
REPRESENTATION=ilg
DOMAIN=ferry
WL=2gwl
ITERATIONS=1

SAVEFILE=tests/${WL}_${MODEL}_${DOMAIN}.joblib

DOMAINFILE=../benchmarks/ipc2023-learning-benchmarks/$DOMAIN/domain.pddl
PROBLEMFILE=../benchmarks/ipc2023-learning-benchmarks/$DOMAIN/testing/easy/p10.pddl

python3 run_kernel.py $DOMAINFILE $PROBLEMFILE $SAVEFILE
