export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/boost_153/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/OpenBLAS/lib/
#need to update data layer and pooling layer before convert to caffemodel

#python json2prototxt.py --mx-json /cloud_data01/zhengmeisong/FRModels/mx/model_r100_10_07/model-symbol.json --cf-prototxt /cloud_data01/zhengmeisong/FRModels/mx/model_r100_10_07/r100.prototxt
# python mxnet2caffe.py --mx-model /cloud_data01/zhengmeisong/FRModels/mx/model_r100_10_07/model --mx-epoch 15 --cf-prototxt /cloud_data01/zhengmeisong/FRModels/mx/model_r100_10_07/r100.prototxt --cf-model /cloud_data01/zhengmeisong/FRModels/mx/model_r100_10_07/r100.caffemodel

python json2prototxt.py --mx-json /cloud_data01/zhengmeisong/FRModels/mx/m2_tri_11_11/model-symbol.json --cf-prototxt /cloud_data01/zhengmeisong/FRModels/mx/m2_tri_11_11/m2.prototxt
#python mxnet2caffe.py --mx-model /cloud_data01/zhengmeisong/FRModels/mx/m2_tri_11_11/model --mx-epoch 17 --cf-prototxt /cloud_data01/zhengmeisong/FRModels/mx/m2_tri_11_11/m2.prototxt --cf-model /cloud_data01/zhengmeisong/FRModels/mx/m2_tri_11_11/m2.caffemodel
