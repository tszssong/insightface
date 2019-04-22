CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network m1 --loss softmax --dataset emore 2>&1 | tee ../../logs/m1.log
