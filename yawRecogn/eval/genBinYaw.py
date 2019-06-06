import argparse
import cv2
import pickle
import numpy as np
import sys
import os

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


parser = argparse.ArgumentParser(description='Package eval images')
# general
parser.add_argument('--data-dir', default='/home/ubuntu/zms/wkspace/FR/myDream/data/CFP/', help='')
parser.add_argument('--image-size', type=int, default=112, help='')
parser.add_argument('--output', default='./cfp_fp_yaw.bin', help='path to save.')
args = parser.parse_args()
ori_img_shape = (218,178,3)


bins = []
issame_list = []
pp = 0
f_list = []
p_list = []
for line in open(args.data_dir+'/protocol/frontal_list_nonli.txt', 'r'):
  pp+=1
  if pp%100==0:
    print('processing', pp)
  line = line.strip().split()
  assert len(line)==2
  f_list.append(line)

for line in open(args.data_dir+'/protocol/profile_list_nonli.txt', 'r'):
  pp+=1
  if pp%100==0:
    print('processing', pp)
  line = line.strip().split()
  assert len(line)==2
  p_list.append(line)

print(len(f_list), len(p_list))

for split_id in range(10):
    split_name = str(split_id+1)
    if len(split_name)<2: 
      split_name = '0'+split_name
    for idx, pair_file in enumerate(['diff.txt', 'same.txt']):
        is_same = True    #diff.txt=0, same.txt=1
        if idx == 0:
          is_same = False
        issame_list.append(is_same)
        full_pair_file = '/home/ubuntu/zms/data/cfp/cfp-dataset/Protocol/Split/FP/'+split_name+'/'+pair_file
        with open(full_pair_file, 'r') as in_f:
          for line in in_f:
            record = line.strip().split(',')
            pair1, pair2 = int(record[0]),int(record[1])
            f_path, p_path = f_list[pair1-1][0], p_list[pair2-1][0]
            f_yaw, p_yaw = f_list[pair1-1][1], p_list[pair2-1][1]
            print(f_path, f_yaw, p_path, p_yaw)
            f_img,p_img = cv2.imread(args.data_dir + f_path),cv2.imread(args.data_dir + p_path)
            print(f_img.shape, p_img.shape)
            assert f_img.shape == ori_img_shape
            assert p_img.shape == ori_img_shape
            f_img_crop, p_img_crop = f_img[80:192,32:144,:], p_img[80:192,32:144,:]
            cv2.imshow("f",f_img_crop)
            cv2.imshow("p",p_img_crop)
            cv2.waitKey(1)
            for im in [f_img_crop, p_img_crop]:
              _, s = cv2.imencode('.jpg', im)
              bins.append(s)

with open(args.output, 'wb') as f:
  pickle.dump((bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)

