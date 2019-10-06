import glob
import numpy

import os
SDPath =  '/data3/zhengmeisong/data/jrPairs/'
FDSets = ['imgs/','xxx']

fid = open('jr.lst','w')
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

fid.close()

