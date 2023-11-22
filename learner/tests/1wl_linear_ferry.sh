MODEL=linear-svr
REPRESENTATION=ilg
DOMAIN=ferry
WL=1wl
ITERATIONS=5

SAVEFILE=${WL}_${MODEL}_${DOMAIN}.joblib

# python3 train_kernel.py -m $MODEL -r $REPRESENTATION -d $DOMAIN -k $WL -l $ITERATIONS --model-save-file $SAVEFILE

DOMAINFILE=../benchmarks/ipc2023-learning-benchmarks/$DOMAIN/domain.pddl
PROBLEMFILE=../benchmarks/ipc2023-learning-benchmarks/$DOMAIN/testing/medium/p10.pddl

python3 run_kernel.py $DOMAINFILE $PROBLEMFILE $SAVEFILE

# rm $SAVEFILE

echo $DOMAINFILE
echo $PROBLEMFILE
echo ""

cat << EOF
## 22-11-2023 (5 iterations) log:
[t=0.42612s, 25304 KB] Plan cost: 133
[t=0.42612s, 25304 KB] Expanded 741 state(s).
[t=0.42612s, 25304 KB] Evaluated 3409 state(s).
[t=0.42612s, 25304 KB] Generated 21535 state(s).
EOF
