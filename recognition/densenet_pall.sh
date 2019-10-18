CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u train_parall.py --network d201 --loss arcface \
                                       --dataset emore 2>&1 | tee ../../mx-log/d201_`date +'%m_%d-%H_%M_%S'`.log
