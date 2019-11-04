export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
DATA_DIR=/cloud_data01/StrongRootData/TrainData/ms1m_emore/
#DATA_DIR=/cloud_data01/zhengmeisong/TrainData/glintv2_emore_ms1m/
NETWORK=d290
MODELDIR=../../mx-models/${NETWORK}_`date +'%m_%d'`
mkdir $MODELDIR
PREFIX="$MODELDIR/model"

CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --data-dir $DATA_DIR \
 --network "$NETWORK" \
 --loss-type 5  --verbose 2000 \
 --margin-m 0.0 --margin-a 1.0 --margin-b 0.35 --margin-s 60 --lr 0.1 \
 --lr-step 80000,160000,240000,320000 \
 --prefix "$PREFIX" --per-batch-size 30 --color 1 \
 2>&1 | tee ../../mx-logs/$NETWORK_`date +'%m_%d-%H_%M'`.log
