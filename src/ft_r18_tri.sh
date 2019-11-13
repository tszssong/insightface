export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
DATA_DIR=/cloud_data01/zhengmeisong/TrainData/jajrcasia/
NETWORK=r18
MODELDIR=../../mx-models/model_${NETWORK}_`date +'%m_%d'`
mkdir $MODELDIR
PREFIX="$MODELDIR/model"
LRSTEPS='300000,340000,360000'
PRETRAIN=/cloud_data01/zhengmeisong/wkspace/ol8/mx-models/r18_11_04/model,140
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u train_triplet.py --network $NETWORK \
	 --loss-type 1  --triplet-bag-size 72000 --triplet-alpha 0.5 --triplet-max-ap 1.3 \
	 --data-dir $DATA_DIR --images-per-identity 7 \
         --lr 0.005 --lr-steps "$LRSTEPS" \
         --ckpt 2 --verbose 5000 \
         --prefix "$PREFIX" --per-batch-size 150 --mom 0.0 \
	 --pretrained $PRETRAIN  2>&1 | tee ../../mx-logs/$NETWORK_tri_`date +'%m_%d-%H_%M'`.log
