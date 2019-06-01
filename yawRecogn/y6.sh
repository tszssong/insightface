CUDA_VISIBLE_DEVICES='0,1,2' python -u train.py --network y6 --loss arcface --dataset emore 2>&1 | tee ../../log/y6_`date +'%m_%d-%H_%M_%S'`.log
