import glob
import numpy

import os
SDPath =  '//data03/zhengmeisong/data/gl2ms1m_dl23W1f1_150WW1_img/'
FDSets = ['imgs/','xxx']

fid = open('casia_jr.lst','w')
count=0

imgSet = FDSets[0]
GatRoots = glob.glob(SDPath+imgSet+'/*')
for idx2,imgID in enumerate(GatRoots):
    if int_sub<63834274:
        continue
    if idx2%1000==0:
        print(idx2)
    imgs=glob.glob(imgID+'/*.jpg')
    for imgName in imgs:
        fid.write('1\t'+imgName+'\t%d\n'%count)
    count=count+1

SDPath =  '/data03/zhengmeisong/jrData/'
FDSets = ['jr-pairs/','xxx']

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

