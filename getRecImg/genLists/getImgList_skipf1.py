import glob
import numpy

import os
SDPath =  '/data3/zhengmeisong/data/gl2ms1m_dl23W1f1_150WW1_img/'
FDSets = ['imgs/','xxx']

fid = open('gl2_ms1m_dl23WW150_jrPairs.lst','w')
count=0

imgSet = FDSets[0]
GatRoots = glob.glob(SDPath+imgSet+'/*')
for idx2,imgID in enumerate(GatRoots):
    sub_path = imgID.split('/')[-1]
    int_sub = int(sub_path.split('_')[-1])
    if int_sub<63690800:
        continue

    if idx2%1000==0:
        print(idx2)
    imgs=glob.glob(imgID+'/*.jpg')
    for imgName in imgs:
        fid.write('1\t'+imgName+'\t%d\n'%count)
    count=count+1

SDPath =  '/data3/zhengmeisong/data/jrPairs/'
FDSets = ['imgs/','xxx']

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

