
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import PIL
from PIL import Image 
from easydict import EasyDict as edict
import time
import sys
import numpy as np
import argparse
import struct
import cv2
import sklearn
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import torch.nn.functional as F
import torchvision
from torchvision import transforms
sys.path.append( os.path.join( os.path.dirname(__file__),'../backbone/') )
from model_resnet import ResNet_50, ResNet_101, ResNet_152
from model_irse import IR_18, IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from model_m2 import MobileV2

def read_img(image_path):
  img = cv2.imread(image_path, cv2.IMREAD_COLOR)
  return img

test_transform = transforms.Compose([ transforms.ToTensor(), 
                     transforms.Normalize(mean =  [0.5, 0.5, 0.5], std =  [0.5, 0.5, 0.5]), ])

def get_feature(imgs, net, device='cpu'):
  count = len(imgs)
  data = torch.empty(count*2, 3, imgs[0].shape[0], imgs[0].shape[1]) 
  for idx, img in enumerate(imgs):
    img = img[:,:,::-1] #to rgb
    # img = np.transpose( img, (2,0,1) )
    for flipid in [0,1]:
      _img = np.copy(img)
      if flipid==1:
        _img = _img[:,::-1,:]
        # _img =  Image.fromarray(_img)
        # _img.save('1.jpg')
        # _img = test_transform(_img)
      _img =  Image.fromarray(_img)
      
      _img = test_transform(_img)
      data[count*flipid+idx] = _img

  with torch.no_grad():
    batchFea = net(data.to(device))
  #batchFea = F.normalize(batchFea).detach()
  x = batchFea.detach().cpu().numpy()
  embedding = x[0:count,:] + x[count:,:]
  embedding = sklearn.preprocessing.normalize(embedding)
  return embedding


def write_bin(path, feature):
  feature = list(feature)
  with open(path, 'wb') as f:
    f.write(struct.pack('4i', len(feature),1,4,5))
    f.write(struct.pack("%df"%len(feature), *feature))

def get_and_write(buffer, nets, device='cpu'):
  imgs = []
  for k in buffer:
    imgs.append(k[0])
  features = get_feature(imgs, nets, device)
  #print(np.linalg.norm(feature))
  assert features.shape[0]==len(buffer)
  for ik,k in enumerate(buffer):
    out_path = k[1]
    feature = features[ik].flatten()
    write_bin(out_path, feature)

def main(args):

  print(args)
  image_shape = [int(x) for x in args.image_size.split(',')]
  DEVICE =  torch.device("cuda:%d"%(int(args.gpu)) if torch.cuda.is_available() else "cpu")
  print("device:",DEVICE)
  INPUT_SIZE = image_shape[1:]
  ModelName = args.model.split('/')[-2]
  BACKBONE = eval(ModelName)(INPUT_SIZE)
  backbone_load_path = args.model
  if backbone_load_path and os.path.isfile(backbone_load_path):
      print("Loading Backbone Checkpoint '{}'".format(backbone_load_path))
      BACKBONE.load_state_dict(torch.load(backbone_load_path, map_location='cuda:%d'%args.gpu)) 
  else:
      print("No Checkpoint Error!" )
  BACKBONE = BACKBONE.to(DEVICE)
  BACKBONE.eval()

  facescrub_out = os.path.join(args.output, 'facescrub')
  megaface_out = os.path.join(args.output, 'megaface')

  i = 0
  succ = 0
  buffer = []
  for line in open(args.facescrub_lst, 'r'):
    if i%1000==0:
      print("writing fs",i, succ)
    i+=1
    image_path = line.strip()
    _path = image_path.split('/')
    a,b = _path[-2], _path[-1]
    out_dir = os.path.join(facescrub_out, a)
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
    image_path = os.path.join(args.facescrub_root, image_path)
    img = read_img(image_path)
    if img is None:
      print('read error:', image_path)
      continue
    out_path = os.path.join(out_dir, b+"_%s.bin"%(args.algo))
    item = (img, out_path)
    buffer.append(item)
    if len(buffer)==args.batch_size:
      get_and_write(buffer, BACKBONE, device=DEVICE)
      buffer = []
    succ+=1
  if len(buffer)>0:
    get_and_write(buffer, BACKBONE, device=DEVICE)
    buffer = []
  print('fs stat',i, succ)

  i = 0
  succ = 0
  buffer = []
  for line in open(args.megaface_lst, 'r'):
    if i%1000==0:
      print("writing mf",i, succ)
    i+=1
    image_path = line.strip()
    _path = image_path.split('/')
    a1, a2, b = _path[-3], _path[-2], _path[-1]
    out_dir = os.path.join(megaface_out, a1, a2)
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
      #continue
    #print(landmark)
    image_path = os.path.join(args.megaface_root, image_path)
    img = read_img(image_path)
    if img is None:
      print('read error:', image_path)
      continue
    out_path = os.path.join(out_dir, b+"_%s.bin"%(args.algo))
    item = (img, out_path)
    buffer.append(item)
    if len(buffer)==args.batch_size:
      get_and_write(buffer, BACKBONE, device=DEVICE)
      buffer = []
    succ+=1
  if len(buffer)>0:
    get_and_write(buffer, BACKBONE, device=DEVICE)
    buffer = []
  print('mf stat',i, succ)


def parse_arguments(argv):
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--batch_size', type=int, help='', default=8)
  parser.add_argument('--image_size', type=str, help='', default='3,112,112')
  parser.add_argument('--gpu', type=int, help='', default=0)
  parser.add_argument('--algo', type=str, help='', default='py')
  parser.add_argument('--facescrub-lst', type=str, help='', default='./data/facescrub_lst')
  parser.add_argument('--megaface-lst', type=str, help='', default='./data/megaface_lst')
  parser.add_argument('--facescrub-root', type=str, help='', default='./data/facescrub_images')
  parser.add_argument('--megaface-root', type=str, help='', default='./data/megaface_images')
  parser.add_argument('--output', type=str, help='', default='./feature_out/')
  parser.add_argument('--model', type=str, help='', default='./IR_50/IR_50_E_23_B1639361_T1912290832.pth')
  return parser.parse_args(argv)

if __name__ == '__main__':
  main(parse_arguments(sys.argv[1:]))

