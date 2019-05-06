CUDA_VISIBLE_DEVICES='0,1' python -u train.py --network r50 --loss arcface --dataset emore 2>&1 | tee /home/zhengmeisong/log.log
