MODEL=blr
REPRESENTATION=ilg
DOMAIN=ferry
WL=1wl
ITERATIONS=4

python3 tests/custom.py -m $MODEL -r $REPRESENTATION -d $DOMAIN -k $WL -l $ITERATIONS --train
