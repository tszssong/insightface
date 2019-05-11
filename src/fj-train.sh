DATA_DIR=/cloud_data01/zhengmeisong/TrainData/glintv2_emore_ms1m/

NETWORK=y6
JOB=loss4-GDC-128-emore
MODELDIR="../../models"
PREFIX="$MODELDIR/model"
CUDA_VISIBLE_DEVICES='2,3' python -u train_softmax.py --data-dir $DATA_DIR \
                                                      --network "$NETWORK" \
                                                      --loss-type 4 \
                                                      --margin-m 0.5 --lr 0.1 \
                                                      --lr-step 80000,160000,240000,320000 \
                                                      --version-output GDC --emb-size 128 \
                                                      --prefix "$PREFIX" --per-batch-size 192 --color 1 2>&1 | tee ../../log/fj.log
