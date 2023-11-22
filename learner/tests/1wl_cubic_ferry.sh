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

echo ""

cat << EOF
## 21-11-2023 (1 iteration) log:
[t=1.09132s, 5393456 KB] Plan cost: 26
[t=1.09132s, 5393456 KB] Expanded 85 state(s).
[t=1.09132s, 5393456 KB] Evaluated 208 state(s).
[t=1.09132s, 5393456 KB] Generated 668 state(s).
EOF

echo ""

cat << EOF
## 14-09-2023 (1 iteration) log:
[t=2.85695s, 3191948 KB] Plan cost: 26
[t=2.85695s, 3191948 KB] Expanded 96 state(s).
[t=2.85695s, 3191948 KB] Evaluated 256 state(s).
[t=2.85695s, 3191948 KB] Generated 757 state(s).
EOF
