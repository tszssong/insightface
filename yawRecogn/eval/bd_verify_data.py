"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import sys
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
import sklearn
import cv2
import math
import datetime
import pickle
from sklearn.decomposition import PCA
import mxnet as mx
from mxnet import ndarray as nd



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='do verification')
  # general
  parser.add_argument('--data-dir', default='/data03/zhengmeisong/TrainData/glintv2_angle/', help='')
  parser.add_argument('--model', default='../models/y6-arcface-emore/model,254', help='path to load model.')
  parser.add_argument('--ftname', default='.ft', type=str)
  parser.add_argument('--target', default='cfp_fp_yaw', help='test targets.')
  parser.add_argument('--gpu', default=0, type=int, help='gpu id')
  parser.add_argument('--batch-size', default=32, type=int, help='')
  parser.add_argument('--max', default='', type=str, help='')
  parser.add_argument('--mode', default=0, type=int, help='')
  parser.add_argument('--nfolds', default=10, type=int, help='')
  args = parser.parse_args()
  sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
  image_size = (112,112,3)
  print('image_size', image_size)
  ctx = mx.gpu(args.gpu)
  # nets = []
  prefix,epoch = args.model.split(',')
  
  time0 = datetime.datetime.now()
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, int(epoch))
  all_layers = sym.get_internals()
  sym = all_layers['fc1_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  # nets.append(model)
  time_now = datetime.datetime.now()
  diff = time_now - time0
  print('model loading time', diff.total_seconds())
  # with open('/data03/zhengmeisong/wkspace/FR/doorbell/logs/ai_img_v4_11.lst','r') as fp:
  with open('/data03/zhengmeisong/wkspace/FR/doorbell/logs/ai_all_smallface_picked.lst','r') as fp:
    lines = fp.readlines()
    for line in lines:
      img_path, label = line.strip().split(' ')
      img = cv2.imread(img_path)
      info_path = img_path.replace('.jpg', '.info')
    
      assert os.path.isfile(info_path)
      with open(info_path) as f:
        info = f.readlines()
      yaw = info[0].strip().split(',')[1]
      yaw = float(yaw)
      yaw = np.array(yaw).reshape(1,-1)
      yaw = nd.array(yaw)
      if not(img.shape[0]==image_size[0] and img.shape[1]==image_size[1]) :
        img = cv2.resize(image_size)
      img = nd.array(img)
      img = nd.transpose(img, axes=(2, 0, 1))

      print(img_path, img.shape, yaw)
      
      in_img = nd.expand_dims(img, axis=0)
      db = mx.io.DataBatch(data=(in_img,yaw))
      model.forward(db, is_train=False)
      net_out = model.get_outputs()
      _embeddings = net_out[0].asnumpy()
      fea = sklearn.preprocessing.normalize(_embeddings).flatten()
      # print(fea)
      fea_path = img_path.replace('.jpg',args.ftname)
      np.savetxt(fea_path, fea)


