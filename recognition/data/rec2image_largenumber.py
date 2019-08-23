from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import mxnet as mx
from mxnet import ndarray as nd
import random
import argparse
import cv2
import time
import sklearn
import numpy as np
import struct
from collections import namedtuple

_IR_FORMAT = 'IfQQ'
_IR_SIZE = struct.calcsize(_IR_FORMAT)
IRHeader = namedtuple('HEADER', ['flag', 'label', 'id', 'id2'])
def my_unpack(s):
    header = IRHeader(*struct.unpack(_IR_FORMAT, s[:_IR_SIZE]))
    s = s[_IR_SIZE:]
    if header.flag > 0:
        header = header._replace(label=np.frombuffer(s, np.double, header.flag))
        #print header.flag
        s = s[header.flag*8:]
    return header, s

def main(args):
  include_datasets = args.include.split(',')
  rec_list = []
  for ds in include_datasets:
    path_imgrec = os.path.join(ds, 'train.rec')
    path_imgidx = os.path.join(ds, 'train.idx')
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
    rec_list.append(imgrec)
  if not os.path.exists(args.output):
    os.makedirs(args.output)
  for ds_id in range(len(rec_list)):
    id_list = []
    imgrec = rec_list[ds_id]
    s = imgrec.read_idx(0)
    header, _ = my_unpack(s)
    assert header.flag>0
    print('header0 label', header.label)
    header0 = (int(header.label[0]), int(header.label[1]))
    seq_identity = range(int(header.label[0]), int(header.label[1]))
    pp=0
    nullid = 0
    for identity in seq_identity:
      id_dir = os.path.join(args.output, "%d_%d"%(ds_id, identity))
      os.makedirs(id_dir)
      pp+=1
      if pp%10000==0:
        print('processing id', pp)
      s = imgrec.read_idx(identity)
      header, _ = my_unpack(s)
      imgid = 0
      for _idx in range(int(header.label[0]), int(header.label[1])):
        s = imgrec.read_idx(_idx)
        _header, _img = my_unpack(s)
        if _img == '':
          nullid = nullid + 1
          print("null:", nullid, imgid)
          continue
        #_header, _img = my_unpack(s)
        _img = mx.image.imdecode(_img).asnumpy()[:,:,::-1] # to bgr
        image_path = os.path.join(id_dir, "%d.jpg"%imgid)
        cv2.imwrite(image_path, _img)
    print("Done! total:",pp,"null:", nullid)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='do dataset merge')
  # general
  parser.add_argument('--include', default='', type=str, help='')
  parser.add_argument('--output', default='', type=str, help='')
  args = parser.parse_args()
  main(args)

