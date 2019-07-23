LRSTEPS='100000,140000,160000'
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network y6 \
 --loss arcface --dataset emore \
 --lr 0.01 --lr-steps "$LRSTEPS" \
 --pretrained './models/model' --pretrained-epoch 50 \
 2>&1 | tee ../../log/y6_`date +'%m_%d-%H_%M_%S'`.log
