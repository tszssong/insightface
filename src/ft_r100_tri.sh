export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
DATA_DIR=/data03/zhengmeisong/TrainData/casia/
MODELDIR=../../model_r100_`date +'%m_%d'`
mkdir $MODELDIR
PREFIX="$MODELDIR/model-r100-triplet"

LRSTEPS='100000,140000,160000'
CUDA_VISIBLE_DEVICES='2,3' python -u train_triplet.py --network r100 \
 --loss-type 1  --triplet-bag-size 72000 --triplet-alpha 0.5 --triplet-max-ap 1.3 \
 --data-dir $DATA_DIR \
 --lr 0.005 --lr-steps "$LRSTEPS" \
 --ckpt 2 \
 --prefix "$PREFIX" --per-batch-size 120 --mom 0.0 \
 --pretrained ../../model-r100-ii/model,0  2>&1 | tee ../../log/r100_tri_`date +'%m_%d-%H_%M_%S'`.log
