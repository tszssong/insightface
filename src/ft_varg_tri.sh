export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
DATA_DIR=/mnt/sdc/zhengmeisong/TrainData/jajrcasia/
MODELDIR=../../mx-models/model_r18_`date +'%m_%d'`
NETWORK=v1
mkdir $MODELDIR
PREFIX="$MODELDIR/model"
LRSTEPS='300000,340000,360000'
CUDA_VISIBLE_DEVICES='0,1' python -u train_triplet.py --network $NETWORK \
	 --loss-type 1  --triplet-bag-size 72000 --triplet-alpha 0.5 --triplet-max-ap 1.3 \
	 --data-dir $DATA_DIR --images-per-identity 7 \
         --lr 0.005 --lr-steps "$LRSTEPS" \
         --ckpt 2 --verbose 5000 \
         --prefix "$PREFIX" --per-batch-size 360 --mom 0.0 \
	 --pretrained ../../mx-models/vargfacenet-arcface-retina/model,142  2>&1 | tee ../../mx-logs/$NETWORK_tri_`date +'%m_%d-%H_%M'`.log

#export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
#DATA_DIR=/mnt/sdc/zhengmeisong/TrainData/jajrcasia/
#MODELDIR=../../mx-models/model_r18_`date +'%m_%d'`
#mkdir $MODELDIR
#PREFIX="$MODELDIR/model"
#LRSTEPS='100000,140000,160000'
#CUDA_VISIBLE_DEVICES='0,1' python -u train_triplet.py --network r18 \
#	 --loss-type 1  --triplet-bag-size 72000 --triplet-alpha 0.5 --triplet-max-ap 1.3 \
#	 --data-dir $DATA_DIR \
#         --lr 0.005 --lr-steps "$LRSTEPS" \
#         --ckpt 2 --verbose 5000 \
#         --prefix "$PREFIX" --per-batch-size 240 --mom 0.0 \
#	 --pretrained ../../mx-preTrain/r18-fj/model,125  2>&1 | tee ../../mx-logs/r18_tri_`date +'%m_%d-%H_%M'`.log
#
