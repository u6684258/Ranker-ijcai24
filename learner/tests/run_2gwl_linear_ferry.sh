MODEL=linear-svr
REPRESENTATION=ilg
DOMAIN=ferry
WL=2gwl
ITERATIONS=1

python3 tests/custom.py -m $MODEL -r $REPRESENTATION -d $DOMAIN -k $WL -l $ITERATIONS --run
