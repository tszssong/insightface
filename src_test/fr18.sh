DATA_DIR=/cloud_data01/StrongRootData/TrainData/glintv2_emore_ms1m/
MODELDIR='../../model_'`date +'%m_%d'`
mkdir $MODELDIR
PREFIX="$MODELDIR/model"

LRSTEPS='100000,140000,160000'
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network f18 \
 --loss-type 4 \
 --data-dir $DATA_DIR \
 --lr 0.01 --lr-steps "$LRSTEPS" \
 --margin-s 64.0 --margin-m 0.5 \
 --version-output GNAP --emb-size 128 \
 --fc7-wd-mult 10.0 --wd 0.00004 \
 --prefix "$PREFIX" --per-batch-size 128 --color 1 \
 | tee ../../log/y6_`date +'%m_%d-%H_%M_%S'`.log
