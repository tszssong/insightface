DATA_DIR=/cloud_data01/zhengmeisong/TrainData/glintv2_emore_ms1m/

NETWORK=d121
MODELDIR="../../mx-models"
PREFIX="$MODELDIR/model"
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u train_softmax.py --data-dir $DATA_DIR \
 --network "$NETWORK" \
 --loss-type 4 \
 --margin-m 0.5 --lr 0.1 \
 --lr-step 80000,160000,240000,320000 \
 --prefix "$PREFIX" --per-batch-size 30 --color 1 \
 2>&1 | tee ../../mx-log/$NETWORK_`date +'%m_%d-%H_%M_%S'`.log
