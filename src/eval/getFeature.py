import os,sys
os.environ['GLOG_minloglevel'] = '2'
import argparse
import cv2
import numpy as np
import glob
from skimage import exposure
import sklearn
from sklearn import preprocessing
import mxnet as mx
from mxnet import ndarray as nd

parser = argparse.ArgumentParser(description='get model')
parser.add_argument('--image-size', default='112,112', help='')
model_str = "/data03/zhengmeisong/wkspace/FR/model-r100-ii/model,0"
model_str = "../../../model/model-r50-triplet-dl23f1-tagv2/model,106"
model_str = "../../../model_r100_09_30/model-r100-triplet,87"
parser.add_argument('--model', default=model_str, help='path to load model.')
parser.add_argument('--gpu', default=1, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 2 means using R+O, else using O')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--ftname', type=str, default='.ft')
#parser.add_argument('--imlist', type=str, default='../../../../../TestData/CASIA-IvS-Test/CASIA-IvS-Test-final-v3-revised.lst')
parser.add_argument('--imlist', type=str, default='/data03/zhengmeisong/TestData/JA-Test/imgs.lst')

args = parser.parse_args()

def get_feature_model(imglist, modelpath, image_size):
    _vec = args.model.split(',')
    assert len(_vec)==2
    prefix = _vec[0]
    epoch = int(_vec[1])
    ctx = mx.gpu(args.gpu)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    saveRoot = prefix + "-%04d"%epoch +'/'
    if not os.path.isdir(saveRoot):
        os.makedirs(saveRoot)

    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    with open(imglist) as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            imgPath = line.strip().split(' ')[0]
            img = cv2.imread(imgPath)
            try:
                img.shape
            except:
                print(imgPath, "not exist!")
                continue
            
            assert img.shape[0] == image_size[0]
            nimg2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            aligned = np.transpose(nimg2, (2,0,1))
            input_blob = np.expand_dims(aligned, axis=0)
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data,))
            model.forward(db, is_train=False)
            _embedding = model.get_outputs()[0].asnumpy()
            embedding =  sklearn.preprocessing.normalize(_embedding) #.flatten()
            # print(embedding.size)
            subRoot = imgPath.split('/')[-2] + '/'
            # print(subRoot)
            if not os.path.isdir(saveRoot+subRoot):
                os.makedirs(saveRoot+subRoot)
            feaPath = saveRoot+subRoot+imgPath.split('/')[-1].replace('.jpg',args.ftname)
            np.savetxt(feaPath, embedding)
            count += 1
            if count % 100 == 0:
                print('%d. '%count),
                # print('.'),
                sys.stdout.flush()

if __name__=='__main__':
    image_size = [int(i) for i in args.image_size.split(',')]
    get_feature_model(args.imlist, args.model, image_size)
