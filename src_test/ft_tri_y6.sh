DATA_DIR=/home/zhengmeisong/TrainData/glintv2_emore_ms1m/
MODELDIR='../../model_'`date +'%m_%d'`
mkdir $MODELDIR
PREFIX="$MODELDIR/model-y6-triplet"

LRSTEPS='100000,140000,160000'
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_triplet.py --network y6 \
 --loss-type 1 \
 --data-dir $DATA_DIR \
 --lr 0.005 --lr-steps "$LRSTEPS" \
 --version-output GNAP --emb-size 128 \
 --prefix "$PREFIX" --per-batch-size 150 --mom 0.0 \
 --pretrained '../../model_05_26/model,178'  | tee ../../log/y6_`date +'%m_%d-%H_%M_%S'`.log
