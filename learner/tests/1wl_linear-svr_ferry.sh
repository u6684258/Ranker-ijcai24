MODEL=linear-svr
REPRESENTATION=ilg
DOMAIN=ferry
WL=1wl

SAVEFILE=${WL}_${MODEL}_${DOMAIN}.joblib

python3 train_kernel.py -m $MODEL -r $REPRESENTATION -d $DOMAIN -k $WL --model-save-file $SAVEFILE

DOMAINFILE=../benchmarks/ipc2023-learning-benchmarks/ferry/domain.pddl
PROBLEMFILE=../benchmarks/ipc2023-learning-benchmarks/ferry/testing/medium/p10.pddl

python3 run_kernel.py $DOMAINFILE $PROBLEMFILE $SAVEFILE

rm $SAVEFILE

### 21-11-2023 log
# [t=0.416947s, 25492 KB] Plan length: 132 step(s).
# [t=0.416947s, 25492 KB] Plan cost: 132
# [t=0.416947s, 25492 KB] Expanded 659 state(s).
# [t=0.416947s, 25492 KB] Reopened 0 state(s).
# [t=0.416947s, 25492 KB] Evaluated 3308 state(s).
# [t=0.416947s, 25492 KB] Evaluations: 3308
# [t=0.416947s, 25492 KB] Generated 19165 state(s).
# [t=0.416947s, 25492 KB] Dead ends: 0 state(s).
# [t=0.416947s, 25492 KB] Number of registered states: 3308
# [t=0.416947s, 25492 KB] Int hash set load factor: 3308/4096 = 0.807617
# [t=0.416947s, 25492 KB] Int hash set resizes: 12
# [t=0.416947s, 25492 KB] Search time: 0.363274s
# [t=0.416947s, 25492 KB] Total time: 0.416947s

### 14-09-2023 log
# [t=0.401167s, 33524 KB] Plan length: 136 step(s).
# [t=0.401167s, 33524 KB] Plan cost: 136
# [t=0.401167s, 33524 KB] Expanded 556 state(s).
# [t=0.401167s, 33524 KB] Reopened 0 state(s).
# [t=0.401167s, 33524 KB] Evaluated 3212 state(s).
# [t=0.401167s, 33524 KB] Evaluations: 3212
# [t=0.401167s, 33524 KB] Generated 16185 state(s).
# [t=0.401167s, 33524 KB] Dead ends: 0 state(s).
# [t=0.401167s, 33524 KB] Number of registered states: 3212
# [t=0.401167s, 33524 KB] Int hash set load factor: 3212/4096 = 0.78418
# [t=0.401167s, 33524 KB] Int hash set resizes: 12
# [Computation by walltime] Search time: 0.395s
