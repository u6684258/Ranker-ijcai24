MODEL=cubic-svr
REPRESENTATION=ilg
DOMAIN=ferry
WL=1wl
ITERATIONS=3

python3 tests/custom.py -m $MODEL -r $REPRESENTATION -d $DOMAIN -k $WL -l $ITERATIONS --train
