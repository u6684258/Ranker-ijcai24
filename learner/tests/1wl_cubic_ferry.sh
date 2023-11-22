MODEL=cubic-svr
REPRESENTATION=ilg
DOMAIN=ferry
WL=1wl
ITERATIONS=1

SAVEFILE=${WL}_${MODEL}_${DOMAIN}.joblib

python3 train_kernel.py -m $MODEL -r $REPRESENTATION -d $DOMAIN -k $WL -l $ITERATIONS --model-save-file $SAVEFILE

DOMAINFILE=../benchmarks/ipc2023-learning-benchmarks/$DOMAIN/domain.pddl
PROBLEMFILE=../benchmarks/ipc2023-learning-benchmarks/$DOMAIN/testing/easy/p10.pddl

python3 run_kernel.py $DOMAINFILE $PROBLEMFILE $SAVEFILE

rm $SAVEFILE

echo $DOMAINFILE
echo $PROBLEMFILE
echo ""

cat << EOF
## 22-11-2023 (1 iteration) log:
[t=1.11288s, 5393360 KB] Plan length: 26 step(s).
[t=1.11288s, 5393360 KB] Expanded 93 state(s).
[t=1.11288s, 5393360 KB] Evaluated 224 state(s).
[t=1.11288s, 5393360 KB] Generated 734 state(s).
EOF
