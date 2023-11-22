LOG_DIR=logs/train_kernel

mkdir -p $LOG_DIR

k=1wl
l=1
r=ilg

for m in linear-svr lasso ridge rbf-svr quadratic-svr cubic-svr mlp
do 
    for domain in blocksworld childsnack ferry floortile miconic rovers satellite sokoban spanner transport
    do
        DESC=${domain}_${r}_${k}_${l}_${m}
        SAVE_FILE="trained_kernel_models/${DESC}.joblib"
        if [ ! -f $SAVE_FILE ]; then
            echo python3 train_kernel.py -d $domain -k $k -l $l -r $r -m $m --model-save-file ${SAVE_FILE} '>' $LOG_DIR/${DESC}.log
                 python3 train_kernel.py -d $domain -k $k -l $l -r $r -m $m --model-save-file ${SAVE_FILE} > $LOG_DIR/${DESC}.log
        fi
    done
done
