MODEL=linear-svr
REPRESENTATION=ilg
DOMAIN=$1
WL=1wl
ITERATIONS=4

python3 tests/custom.py -m $MODEL -r $REPRESENTATION -d $DOMAIN -k $WL -l $ITERATIONS --run --online
