MODEL=cubic-svr
REPRESENTATION=ilg
DOMAIN=ferry
WL=1wl
ITERATIONS=1

SAVEFILE=tests/${WL}_${MODEL}_${DOMAIN}.joblib

python3 train_kernel.py -m $MODEL -r $REPRESENTATION -d $DOMAIN -k $WL -l $ITERATIONS --model-save-file $SAVEFILE
