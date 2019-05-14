from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import logging
import sys
import numbers
import math
import sklearn
import datetime
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio

import copy

logger = logging.getLogger()


class FaceImageIter(io.DataIter):

    def __init__(self, batch_size, data_shape,
                 path_imgrec = None,
                 shuffle=False, aug_list=None, mean = None,
                 rand_mirror = False, cutoff = 0, color_jittering = 0,
                 images_filter = 0,
                 data_name='data', label_name='softmax_label', **kwargs):
        super(FaceImageIter, self).__init__()
        assert path_imgrec
        if path_imgrec:
            logging.info('loading recordio %s...',
                         path_imgrec)
            path_imgidx = path_imgrec[0:-4]+".idx"
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
            s = self.imgrec.read_idx(0)
            header, _ = recordio.unpack(s)
            if header.flag>0:
              print('header0 label', header.label)
              self.header0 = (int(header.label[0]), int(header.label[1]))
              #assert(header.flag==1)
              #self.imgidx = range(1, int(header.label[0]))
              self.imgidx = []
              self.id2range = {}
              self.seq_identity = range(int(header.label[0]), int(header.label[1]))
              for identity in self.seq_identity:
                s = self.imgrec.read_idx(identity)
                header, _ = recordio.unpack(s)
                a,b = int(header.label[0]), int(header.label[1])
                count = b-a
                if count<images_filter:
                  continue
                self.id2range[identity] = (a,b)
                self.imgidx += range(a, b)
              print('id2range', len(self.id2range))
              print('total id', len(self.imgidx))
            else:
              self.imgidx = list(self.imgrec.keys)
            if shuffle:
              self.seq = self.imgidx
              self.oseq = self.imgidx
              print(len(self.seq))
            else:
              self.seq = None

        self.mean = mean
        self.nd_mean = None
        if self.mean:
          self.mean = np.array(self.mean, dtype=np.float32).reshape(1,1,3)
          self.nd_mean = mx.nd.array(self.mean).reshape((1,1,3))

        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.image_size = '%d,%d'%(data_shape[1],data_shape[2])
        self.rand_mirror = rand_mirror
        print('rand_mirror', rand_mirror)
        self.cutoff = cutoff
        self.color_jittering = color_jittering
        self.CJA = mx.image.ColorJitterAug(0.125, 0.125, 0.125)
        self.provide_label = [(label_name, (batch_size,))]
        print(self.provide_label[0][1])
        self.cur = 0
        self.nbatch = 0
        self.is_init = False

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        print('call reset()')
        self.cur = 0
        if self.shuffle:
          random.shuffle(self.seq)
        if self.seq is None and self.imgrec is not None:
            self.imgrec.reset()

    def num_samples(self):
      return len(self.seq)

    def next_sample(self):
        """Helper function for reading in next sample."""
        #set total batch size, for example, 1800, and maximum size for each people, for example 45
        if self.seq is not None:
          while True:
            if self.cur >= len(self.seq):
                raise StopIteration
            idx = self.seq[self.cur]
            self.cur += 1
            if self.imgrec is not None:
              s = self.imgrec.read_idx(idx)
              header, img = recordio.unpack(s)
              label = header.label
              if not isinstance(label, numbers.Number):
                label = label[0]
                angle = label[3]
              print('1',label)
              return label, img, angle, None, None
            else:
              label, fname, bbox, landmark = self.imglist[idx]
              print('2[ds]:not support yet!',label)
              return label, self.read_image(fname), bbox, landmark
        else:
            s = self.imgrec.read()
            if s is None:
                raise StopIteration
            header, img = recordio.unpack(s)
#            print('3',header.label)
            return header.label[0], img, header.label[3], None, None

    def brightness_aug(self, src, x):
      alpha = 1.0 + random.uniform(-x, x)
      src *= alpha
      return src

    def contrast_aug(self, src, x):
      alpha = 1.0 + random.uniform(-x, x)
      coef = nd.array([[[0.299, 0.587, 0.114]]])
      gray = src * coef
      gray = (3.0 * (1.0 - alpha) / gray.size) * nd.sum(gray)
      src *= alpha
      src += gray
      return src

    def saturation_aug(self, src, x):
      alpha = 1.0 + random.uniform(-x, x)
      coef = nd.array([[[0.299, 0.587, 0.114]]])
      gray = src * coef
      gray = nd.sum(gray, axis=2, keepdims=True)
      gray *= (1.0 - alpha)
      src *= alpha
      src += gray
      return src

    def color_aug(self, img, x):
      #augs = [self.brightness_aug, self.contrast_aug, self.saturation_aug]
      #random.shuffle(augs)
      #for aug in augs:
      #  #print(img.shape)
      #  img = aug(img, x)
      #  #print(img.shape)
      #return img
      return self.CJA(img)

    def mirror_aug(self, img):
      _rd = random.randint(0,1)
      if _rd==1:
        for c in xrange(img.shape[2]):
          img[:,:,c] = np.fliplr(img[:,:,c])
      return img

    def compress_aug(self, img):
      buf = BytesIO()
      img = Image.fromarray(img.asnumpy(), 'RGB')
      q = random.randint(2, 20)
      img.save(buf, format='JPEG', quality=q)
      buf = buf.getvalue()
      img = Image.open(BytesIO(buf))
      return nd.array(np.asarray(img, 'float32'))


    def next(self):
        if not self.is_init:
          self.reset()
          self.is_init = True
        """Returns the next batch of data."""
        #print('in next', self.cur, self.labelcur)
        self.nbatch+=1
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        if self.provide_label is not None:
          rows = self.provide_label[0][1][0] #batch_size
          batch_label = nd.empty([rows,2])
          #batch_label = nd.empty([self.provide_label[0][1],2])
        i = 0
        try:
            while i < batch_size:
                label, s, angle, bbox, landmark = self.next_sample()
                _data = self.imdecode(s)
                if _data.shape[0]!=self.data_shape[1]:
                  _data = mx.image.resize_short(_data, self.data_shape[1])
                if self.rand_mirror:
                  _rd = random.randint(0,1)
                  if _rd==1:
                    _data = mx.ndarray.flip(data=_data, axis=1)
                if self.color_jittering>0:
                  if self.color_jittering>1:
                    _rd = random.randint(0,1)
                    if _rd==1:
                      _data = self.compress_aug(_data)
                  #print('do color aug')
                  _data = _data.astype('float32', copy=False)
                  #print(_data.__class__)
                  _data = self.color_aug(_data, 0.125)
                if self.nd_mean is not None:
                  _data = _data.astype('float32', copy=False)
                  _data -= self.nd_mean
                  _data *= 0.0078125
                if self.cutoff>0:
                  _rd = random.randint(0,1)
                  if _rd==1:
                    #print('do cutoff aug', self.cutoff)
                    centerh = random.randint(0, _data.shape[0]-1)
                    centerw = random.randint(0, _data.shape[1]-1)
                    half = self.cutoff//2
                    starth = max(0, centerh-half)
                    endh = min(_data.shape[0], centerh+half)
                    startw = max(0, centerw-half)
                    endw = min(_data.shape[1], centerw+half)
                    #print(starth, endh, startw, endw, _data.shape)
                    _data[starth:endh, startw:endw, :] = 128
                data = [_data]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                for datum in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = [label, angle]
                    i += 1
        except StopIteration:
            if i<batch_size:
                raise StopIteration

        return io.DataBatch([batch_data], [batch_label], batch_size - i)

    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')

    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')

    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        img = mx.image.imdecode(s) #mx.ndarray
        return img

    def read_image(self, fname):
        """Reads an input image `fname` and returns the decoded raw bytes.

        Example usage:
        ----------
        >>> dataIter.read_image('Face.jpg') # returns decoded raw bytes.
        """
        with open(os.path.join(self.path_root, fname), 'rb') as fin:
            img = fin.read()
        return img

    def augmentation_transform(self, data):
        """Transforms input data with specified augmentation."""
        for aug in self.auglist:
            data = [ret for src in data for ret in aug(src)]
        return data

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))

class FaceImageIterList(io.DataIter):
  def __init__(self, iter_list):
    assert len(iter_list)>0
    self.provide_data = iter_list[0].provide_data
    self.provide_label = iter_list[0].provide_label
    self.iter_list = iter_list
    self.cur_iter = None

  def reset(self):
    self.cur_iter.reset()

  def next(self):
    self.cur_iter = random.choice(self.iter_list)
    while True:
      try:
        ret = self.cur_iter.next()
      except StopIteration:
        self.cur_iter.reset()
        continue
      return ret

if __name__=='__main__':
    print ("read rec2img")
    save_dir = '../../../TrainData/glintv2_demo/data/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    train_dataiter = FaceImageIter(
          batch_size           = 1,
          data_shape           = (3, 112, 112),
          path_imgrec          = '../../../TrainData/glintv2/train.rec',
          shuffle              = False,
          rand_mirror          = False,
          mean                 = None,
          cutoff               = False,
          color_jittering      = 0,
          images_filter        = 0,
    )
    num_of_img = len(train_dataiter.imgidx)
    labelDict = {}
    for i in range(num_of_img):
        batch = train_dataiter.next()
        label = batch.label[0].asnumpy()
        id_label = label[0][0]
        angle = label[0][1]
        image = batch.data[0].asnumpy()
        im = copy.deepcopy(image[0])
        im = im.transpose((1,2,0)).astype(np.uint8)
        im = im[...,::-1]
        sub_dir = 'class_%06d'%int(id_label)
        if not os.path.exists(save_dir+'/'+sub_dir):
            os.mkdir(save_dir+'/'+sub_dir)
        if sub_dir in labelDict:
            labelDict[sub_dir] += 1
        else:
            labelDict[sub_dir] = 0
        im_name = 'img_%04d.jpg'%labelDict[sub_dir]
        # cv2.imwrite(save_dir+'/'+sub_dir+'/'+im_name, im)
        # img = cv2.imread(save_dir+'/'+sub_dir+'/'+im_name)
        img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'id:%d'%(id_label),(2,20), font,0.7,(0,0,255),2)
        cv2.putText(img, 'yaw:%.1f'%(angle),(2,90), font,0.6,(0,0,255),2)
        cv2.imwrite(save_dir+'/'+sub_dir+'/'+im_name, img)

        if i%10000==0:
            print ('%7d of %7d img processed!'%(i,num_of_img))
    print (num_of_img, 'processed')
