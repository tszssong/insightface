DATAROOT=/cloud_data01/StrongRootData/TestData/CASIA-IvS-Test/
for idx in `seq -f '%g' 1 4`
do
  echo $idx
  python getFeature.py --model ../../../mx-model/model_r100_10_06/model,$idx \
                       --imlist /cloud_data01/StrongRootData/TestData/CASIA-IvS-Test/CASIA-IvS-Test-final-v3-revised.lst 
  python test_acc_v4_by_rank.py --ftname '.ft' --model ../../../mx-model/model_r100_10_06/model,$idx \
                             --imgRoot $DATAROOT --listFile CASIA-IvS-Test-final-v3-revised.lst \
                             --saveFP 0
done
