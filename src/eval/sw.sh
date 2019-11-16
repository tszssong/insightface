DATAROOT=/cloud_data01/zhengmeisong/testData/
MODEL=/cloud_data01/zhengmeisong/wkspace/models/mx/model-r50-triplet-dl23f1-tagv2/model,106
MODEL=/cloud_data01/zhengmeisong/wkspace/mx-model/olx/model_r100_10_07/model,15
#MODEL=/cloud_data01/zhengmeisong/hpc40/mx-models/hpc13/model_r18_10_25/model,32
#MODEL=/cloud_data01/zhengmeisong/hpc40/mx-models/hpc39/model_r50_10_25/model,30
#MODEL=/cloud_data01/zhengmeisong/hpc40/mx-models/hpc14/model_m1_10_29/model,66
#MODEL=../../../mx-model/model_r100_10_06/model,6
MODEL=//cloud_data01/zhengmeisong/FRModels/mx/model_r50_11_03/model,32
MODEL=/cloud_data01/zhengmeisong/FRModels/mx/model_r100_10_07/model,15
python getFeature.py --model $MODEL --imlist $DATAROOT/sw/sw1v1_112.txt
python getFeature.py --model $MODEL --imlist $DATAROOT/sw/sw1vn_112.txt
echo $MODEL
python testSW1N.py --model $MODEL --ftSize 512

##test 128 dim
#MODEL=/cloud_data01/zhengmeisong/hpc40/mx-models/hpc36/model_y1_10_28/model,74
#python getFeature.py --model $MODEL --imlist $DATAROOT/sw/sw1v1_112.txt
#python getFeature.py --model $MODEL --imlist $DATAROOT/sw/sw1vn_112.txt
#echo $MODEL
#python testSW1N.py --model $MODEL --ftSize 128
