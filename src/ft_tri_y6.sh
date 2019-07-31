DATA_DIR=/data03/zhengmeisong/TrainData/dl2dl3_washed_v1/
#DATA_DIR=/data03/zhengmeisong/TrainData/glintv2_emore_ms1m_dl23W1f1_150WW1/
MODELDIR=../../model_dl2dl3_`date +'%m_%d'`
mkdir $MODELDIR
PREFIX="$MODELDIR/model-y6-triplet"

LRSTEPS='100000,140000,160000'
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_triplet.py --network y6 \
 --loss-type 1  --triplet-bag-size 72000 --triplet-alpha 0.5 --triplet-max-ap 1.3 \
 --data-dir $DATA_DIR \
 --lr 0.005 --lr-steps "$LRSTEPS" \
 --version-output GNAP --emb-size 128 \
 --prefix "$PREFIX" --per-batch-size 60 --mom 0.0 \
 --pretrained '../../model_05_24/model,68' 2>&1 | tee ../../log/y6_`date +'%m_%d-%H_%M_%S'`.log

