export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
DATA_DIR=/cloud_data01/zhengmeisong/TrainData/jajrcasia/
MODELDIR=../../mx-model/model_r100_`date +'%m_%d'`
mkdir $MODELDIR
PREFIX="$MODELDIR/model"

LRSTEPS='100000,140000,160000'
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u train_triplet.py --network r100 \
 --loss-type 1  --triplet-bag-size 72000 --triplet-alpha 0.5 --triplet-max-ap 1.3 \
 --data-dir $DATA_DIR \
 --lr 0.005 --lr-steps "$LRSTEPS" \
 --ckpt 2 --verbose 5000 \
 --prefix "$PREFIX" --per-batch-size 60 --mom 0.0 \
 --pretrained ../../../model-r100-ii/model,0  2>&1 | tee ../../mx-log/r100_tri_`date +'%m_%d-%H_%M_%S'`.log
