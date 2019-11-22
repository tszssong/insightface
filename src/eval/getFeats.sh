MODEL=../../../mx-preTrain/x101-cy/
#MODEL=../../../mx-preTrain/r200_10_10/
MODEL=../../../mx-preTrain/dpn131/
MODEL=../../../mx-preTrain/model-glore152-fj/
MODEL=/cloud_data01/zhengmeisong/wkspace/models/mx/model-r50-triplet-dl23f1-tagv2/
MODEL=../../../mx-preTrain/r18-fj/
MODEL=../../../mx-preTrain/model-r100-ii/model,0
MODEL=/cloud_data01/zhengmeisong/FRModels/model_r100_10_07/model,15
python getFeatBatch.py --model $MODEL --gpu 1\
                 --imlist /cloud_data01/zhengmeisong/data/ms1m_emore_img/imgs.lst \
                 --batchsize 40 --ft '.r100'

