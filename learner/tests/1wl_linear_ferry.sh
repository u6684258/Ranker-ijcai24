MODEL=linear-svr
REPRESENTATION=ilg
DOMAIN=ferry
WL=1wl
ITERATIONS=5

SAVEFILE=${WL}_${MODEL}_${DOMAIN}.joblib

python3 train_kernel.py -m $MODEL -r $REPRESENTATION -d $DOMAIN -k $WL -l $ITERATIONS --model-save-file $SAVEFILE

DOMAINFILE=../benchmarks/ipc2023-learning-benchmarks/$DOMAIN/domain.pddl
PROBLEMFILE=../benchmarks/ipc2023-learning-benchmarks/$DOMAIN/testing/medium/p10.pddl

python3 run_kernel.py $DOMAINFILE $PROBLEMFILE $SAVEFILE

rm $SAVEFILE

echo $DOMAINFILE
echo $PROBLEMFILE
echo ""

cat << EOF
## 21-11-2023 (5 iterations) log:
[t=0.416947s, 25492 KB] Plan cost: 132
[t=0.416947s, 25492 KB] Expanded 659 state(s).
[t=0.416947s, 25492 KB] Evaluated 3308 state(s).
[t=0.416947s, 25492 KB] Generated 19165 state(s).
EOF

echo ""

cat << EOF
## 14-09-2023 (1 iteration) log:
[t=0.401167s, 33524 KB] Plan cost: 136
[t=0.401167s, 33524 KB] Expanded 556 state(s).
[t=0.401167s, 33524 KB] Evaluated 3212 state(s).
[t=0.401167s, 33524 KB] Generated 16185 state(s).
EOF
