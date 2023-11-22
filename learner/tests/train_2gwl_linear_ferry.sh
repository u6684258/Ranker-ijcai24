MODEL=linear-svr
REPRESENTATION=ilg
DOMAIN=ferry
WL=2gwl
ITERATIONS=1

SAVEFILE=tests/${WL}_${ITERATIONS}_${MODEL}_${DOMAIN}.joblib

python3 train_kernel.py -m $MODEL -r $REPRESENTATION -d $DOMAIN -k $WL -l $ITERATIONS --model-save-file $SAVEFILE
