CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train.py --network r100 --loss triplet --lr 0.005 --pretrained '../../model-r100-ii/model' --pretrained-epoch 0
