import argparse
import cv2
import pickle
import numpy as np
import sys
import os

parser = argparse.ArgumentParser(description='Package eval images')
# general
parser.add_argument('--data-dir', default='/home/ubuntu/zms/TrainData/mxTest/', help='')
parser.add_argument('--image-size', type=int, default=112, help='')
# parser.add_argument('--dataset', default='cfp_fp', help='name to gen.')
# parser.add_argument('--dataset', default='casia_ivs', help='name to gen.')
parser.add_argument('--dataset', default='ja_rd_ivs', help='name to gen.')
# parser.add_argument('--dataset', default='lfw', help='name to gen.')
# parser.add_argument('--dataset', default='agedb_30', help='name to gen.')
args = parser.parse_args()
ori_img_shape = (112,112,3)

bins = []
issame_list = []
yaw_list = []
pp = 0
pic_list = []
for line in open(args.data_dir + args.dataset + '/issame.lst', 'r'):
  pp+=1
  if pp%100==0:
    print('processing', pp)
  line = line.strip().split()
  assert len(line)==2
  pic_list.append(line)
print( len(pic_list) )

for id, is_same in pic_list:
  if is_same == 'True':
    b_same = True
  else:
    b_same = False
  path1 = args.data_dir + args.dataset + '/' + id + '_1'
  path2 = args.data_dir + args.dataset + '/' + id + '_2'
  img1, img2 = cv2.imread(path1+'.jpg'),cv2.imread(path2+'.jpg')
    
  print(id, is_same, b_same)
  assert img1.shape == ori_img_shape
  assert img2.shape == ori_img_shape
  cv2.imshow("img1",img1)
  cv2.imshow("img2",img2)
  cv2.waitKey(1)
  for im in [img1, img2]:
    _, s = cv2.imencode('.jpg', im)
    bins.append(s)
  issame_list.append(b_same)
print(len(bins), len(issame_list))

with open(args.data_dir + args.dataset + '.bin', 'wb') as f:
  pickle.dump((bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
  # pickle.dump((bins, yaw_list,  issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)

