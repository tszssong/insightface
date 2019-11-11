#这个配置下失误把margin-m设成了0.5,直接不收敛 --loss-type 5 --margin-m 0.3 --margin-a 1.0 --margin-b 0.2 --ckpt 2\
DATA_DIR=/data01/users/zhengmeisong/TrainData/glintv2_emore_ms1m/
NETWORK=m2
MODELDIR=../../mx-models/${NETWORK}_`date +'%m_%d'`
mkdir $MODELDIR
PREFIX="$MODELDIR/model"
CUDA_VISIBLE_DEVICES='0,2,3' python -u train_softmax.py --data-dir $DATA_DIR \
 --network "$NETWORK" \
 --loss-type 5 --margin-m 0.3 --margin-a 1.0 --margin-b 0.2 --ckpt 2\
 --lr 0.1 \
 --lr-step 100000,160000,240000,320000 \
 --emb-size 512 \
 --prefix "$PREFIX" --per-batch-size 256 --color 1 \
 2>&1 | tee ../../mx-logs/$NETWORK_`date +'%m_%d-%H_%M_%S'`.log
 

