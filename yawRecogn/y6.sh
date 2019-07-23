CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network y6 --loss softmax --per-batch-size 128 --dataset emore 2>&1 | tee ../../log/y6yaw_`date +'%m_%d-%H_%M_%S'`.log
