CUDA_VISIBLE_DEVICES='0' python -u train.py --network m1 --loss softmax --dataset emore 2>&1 | tee ../../logs/m1_`date +'%m_%d-%H_%M_%S'`.log
