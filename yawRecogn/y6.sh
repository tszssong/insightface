CUDA_VISIBLE_DEVICES='1' python -u train.py --network y6 --loss softmax --dataset emore 2>&1 | tee ../../log/y6_`date +'%m_%d-%H_%M_%S'`.log
