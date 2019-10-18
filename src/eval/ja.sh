DATAROOT=/cloud_data01/StrongRootData/TestData/
for idx in `seq -f '%g' 1 5`
do
  echo $idx
  python getFeature.py --model ../../../mx-model/model_r100_10_06/model,$idx \
                       --imlist /cloud_data01/StrongRootData/TestData/JA-Test/imgs.lst
  python test_acc_ja.py --ftname '.ft' --model ../../../mx-model/model_r100_10_06/model,$idx \
     --imgRoot $DATAROOT/JA-Test/ \
     --idListFile id.lst  --faceListFile face.lst \
     --saveFP 0
done

