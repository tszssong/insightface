DATAROOT=/cloud_data01/zhengmeisong/TestData/JA-Test/
MODEL=../../../mx-models/r50_tri_11_25/
for idx in `seq -f '%g' 9 9`
do
  echo $idx
  python getFeature.py --model $MODEL/model,$idx \
                       --imlist ${DATAROOT}/imgs.lst
  python test_acc_ja.py --ftname '.ft' --model ${MODEL}/model,$idx \
     --imgRoot $DATAROOT --ftSize 128\
     --idListFile id.lst  --faceListFile face.lst \
     --saveFP 0
done

