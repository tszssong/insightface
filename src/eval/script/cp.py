import os, sys
import shutil
dst_root = 'toZH'
if not os.path.isdir(dst_root):
  os.makedirs(dst_root)
with open('2w.lst', 'r') as fr:
  lines = fr.readlines()
  for idx,line in enumerate(lines):
    path = line.strip().split(' ')[0]
    dir_path,jpg_path = os.path.split(path)
    sub_path = os.path.split(dir_path)[-1]
    if idx%1000 == 0:
      print(dir_path)
      print(sub_path)
      print(jpg_path)
    dst_path = os.path.join(dst_root, sub_path) 
    if not os.path.exists(dst_path):
      os.makedirs(dst_path)
    shutil.copy(path, os.path.join(dst_path,jpg_path))
    
