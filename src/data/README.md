## gen rec  
在一个路径下放好train.lst property  
property写入:13117709,112,112  
准备好数据，train.lst内容如下：  
```
1       /data3/zhengmeisong/data/jaTrain/imgs/411502198507300517/ja50wa_010610_face.jpg 0
1       /data3/zhengmeisong/data/jaTrain/imgs/411502198507300517/ja50wa_010610_id.jpg   0
1       /data3/zhengmeisong/data/jaTrain/imgs/133023197710063814/ja50wb_017556_id.jpg   1
```
- python face2rec_ms2.py /data3/zhengmeisong/TrainData/ms/train.lst  
输出如下：  
```
(mx-cpu) [zhengmeisong@test04 data]$ python face2rec_ms.py /data3/zhengmeisong/TrainData/ms/train.lst 
/data3/zhengmeisong/anaconda2/envs/mx-cpu/lib/python2.7/site-packages/mxnet/numpy_op_signature.py:61: UserWarning: Some mxnet.numpy operator signatures may not be displayed consistently with their counterparts in the official NumPy package due to too-low Python version 2.7.17 |Anaconda, Inc.| (default, Oct 21 2019, 19:04:46) 
[GCC 7.3.0]. Python >= 3.5 is required to make the signatures display correctly.
  .format(str(sys.version)))
image_size [112, 112]
Creating .rec file from /data3/zhengmeisong/TrainData/ms/train.lst in /data3/zhengmeisong/TrainData/ms
multiprocessing not available, fall back to single threaded encoding
time: 0.257126092911  count: 0
time: 3.14794301987  count: 1000
time: 0.908530950546  count: 2000
time: 0.820667028427  count: 3000
time: 0.705464839935  count: 4000
time: 0.743786096573  count: 5000
time: 0.670212030411  count: 6000
time: 0.633912086487  count: 7000
time: 0.687039852142  count: 8000
time: 0.512052059174  count: 9000
```
  


