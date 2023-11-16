LOG_DIR=logs/train_kernel

mkdir -p $LOG_DIR

k=wl
r=ig

for iii in "heuristic " "deadend --deadends"
do
    set -- $i # Convert the "tuple" into the param args $1 $2...
    echo $1 and $2
    for l in 1
    do
        for domain in ferry blocksworld childsnack floortile miconic rovers satellite sokoban spanner transport
        do 
            for m in linear-svr lasso ridge rbf-svr quadratic-svr cubic-svr mlp
            do
                SAVE_FILE=${m}_${r}_${domain}_${k}_${l}
                if [ ! -f "trained_models_kernel/${SAVE_FILE}.joblib" ]; then
                    echo python3 train_kernel.py $domain -k $k -l $l -r $r -m $m --save-file ${SAVE_FILE} '>' $LOG_DIR/${SAVE_FILE}.log
                        python3 train_kernel.py $domain -k $k -l $l -r $r -m $m --save-file ${SAVE_FILE} '>' $LOG_DIR/${SAVE_FILE}.log
                fi
            done
        done
    done
