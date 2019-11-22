export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
DATA_DIR=/cloud_data01/StrongRootData/TrainData/glintv2_emore_ms1m/
NETWORK=v1
MODELDIR=../../mx-models/${NETWORK}_`date +'%m_%d'`
mkdir $MODELDIR
PREFIX="$MODELDIR/model"
CUDA_VISIBLE_DEVICES='0,1' python -u train_softmax.py --data-dir $DATA_DIR \
 --network "$NETWORK" \
 --loss-type 5 --margin-m 0.3 --margin-a 1.0 --margin-b 0.2 --ckpt 2\
 --lr 0.1 \
 --lr-step 100000,160000,240000,320000 \
 --emb-size 512 \
 --prefix "$PREFIX" --per-batch-size 256 --color 1 \
 2>&1 | tee ../../mx-logs/${NETWORK}_`date +'%m_%d-%H_%M_%S'`.log

# 以上配置单卡320不收敛
 

