CUDA_VISIBLE_DEVICES='0,1' python -u train.py --network m1 --loss softmax --dataset emore 2>&1 | tee /home/zhengmeisong/log.log
