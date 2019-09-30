Models=/data03/zhengmeisong/wkspace/FR/model/model-G50-triplet-dl23W1f1_150W1_V2/
Models=/data03/zhengmeisong/wkspace/FR/model_r100_09_28/
Models=/data03/zhengmeisong/wkspace/FR/model-r100-ii/
DATROOT1=/data03/zhengmeisong/TestData/CASIA-IvS-Test/
#python test_acc_v4_by_rank.py --ftname '.ft' --model $Models/model,106 --imgRoot $DATROOT1/ --listFile CASIA-IvS-Test-final-v3-revised.lst
python test_acc_v4_by_rank.py --ftname '.ft' --model $Models/model,0 --imgRoot $DATROOT1/ --listFile CASIA-IvS-Test-final-v3-revised.lst
#python test_acc_v4_by_rank.py --ftname '.ft' --model $Models/model-r100-triplet,1 --imgRoot $DATROOT1/ --listFile CASIA-IvS-Test-final-v3-revised.lst

