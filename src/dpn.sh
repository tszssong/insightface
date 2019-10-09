DATA_DIR=/data04/zhengmeisong/TrainData/glintv2_emore_ms1m/
NETWORK=p92
MODELDIR="../../mx-models"
PREFIX="$MODELDIR/model"
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --data-dir $DATA_DIR \
 --network "$NETWORK" \
 --loss-type 4 \
 --margin-m 0.5 --lr 0.1 \
 --lr-step 80000,160000,240000,320000 \
 --emb-size 512 \
 --prefix "$PREFIX" --per-batch-size 64 --color 1 \
 2>&1 | tee ../../mx-log/$NETWORK_`date +'%m_%d-%H_%M_%S'`.log
