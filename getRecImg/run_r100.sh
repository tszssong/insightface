CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train.py --network r100 --loss arcface --dataset emore 2>&1 | tee ../../logs/r100_`date +'%m_%d-%H_%M_%S'`.log
