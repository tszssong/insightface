Models=/data03/zhengmeisong/wkspace/FR/model_r100_09_28
Models=/data03/zhengmeisong/wkspace/FR/model/model-G50-triplet-dl23W1f1_150W1_V2/
DATROOT1=/data03/zhengmeisong/TestData/
python test_acc_ja.py --ftname '.ft' --model $Models/model,106 \
     --imgRoot $DATROOT1/JA-Test/ \
     --idListFile id.lst  --faceListFile face.lst

