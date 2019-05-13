import glob
import numpy

import os
SDPath =  '/media/ubuntu/9a42e1da-25d8-4345-a954-4abeadf1bd02/home/ubuntu/song/data/celebrity/'
FDSets = ['data/','xxx']

fid = open('imgList2.lst','w')
count=0

imgSet = FDSets[0]
GatRoots = glob.glob(SDPath+imgSet+'/*')
for idx2,imgID in enumerate(GatRoots):
    if idx2%1000==0:
        print(idx2)
    imgs=glob.glob(imgID+'/*.jpg')
    for imgName in imgs:
        fid.write('1\t'+imgName+'\t%d\n'%count)
    count=count+1

SDPath =  '/media/ubuntu/9a42e1da-25d8-4345-a954-4abeadf1bd02/home/ubuntu/song/data/ms1m_emore_img/'
FDSets = ['data/','xxx']

imgSet = FDSets[0]
GatRoots = glob.glob(SDPath+imgSet+'/*')
for idx2,imgID in enumerate(GatRoots):
    if idx2%1000==0:
        print(idx2)
    imgs=glob.glob(imgID+'/*.jpg')
    for imgName in imgs:
        fid.write('1\t'+imgName+'\t%d\n'%count)
    count=count+1

fid.close()

