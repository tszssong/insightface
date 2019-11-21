export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
DATA_DIR=/cloud_data01/zhengmeisong/TrainData/jajrcasia/
#DATA_DIR=/cloud_data01/zhengmeisong/TrainData/ms1m_1w/
NETWORK=v1
MODELDIR=../../mx-models/supertri_${NETWORK}_`date +'%m_%d'`
mkdir $MODELDIR
PREFIX="$MODELDIR/model"
LRSTEPS='300000,340000,360000'
#PRETRAIN=../../mx-preTrain/r18_11_04/model,146
PRETRAIN=../../mx-preTrain/vargfacenet/model,18
SUPER=../../mx-preTrain/model_r100_10_07/model,15
CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train_triplet2model.py --network $NETWORK \
	 --loss-type 1  --triplet-bag-size 72000 --triplet-alpha 0.5 --triplet-max-ap 1.3 \
	 --data-dir $DATA_DIR --images-per-identity 7 \
         --lr 0.005 --lr-steps "$LRSTEPS" \
         --ckpt 2 --verbose 10 \
         --supermodel $SUPER \
         --prefix "$PREFIX" --per-batch-size 30 --mom 0.0 \
	 --pretrained $PRETRAIN  2>&1 | tee ../../mx-logs/$NETWORK_tri_`date +'%m_%d-%H_%M'`.log
