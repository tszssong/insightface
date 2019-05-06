import glob
import numpy

import os
SDPath =  '/newdata/liuluoqi/FR/'
FDSets = ['celebrity/','xxx']

fid = open('imgList.lst','w')
count=0

imgSet = FDSets[0]
GatRoots = glob.glob(SDPath+imgSet+'/*')
for idx2,imgID in enumerate(GatRoots):
    if idx2%1000==0:
        print(idx2)
    imgs=glob.glob(imgID+'/*.jpg')
    for imgName in imgs:
        fid.write('1\t'+imgName+'\t%d\t'%count)
        infoName = imgName[:-4]+'.info'
        #print imgName, infoName
        info = numpy.loadtxt(infoName,delimiter=',').astype(numpy.float32)
        if len(info) >=3:
            fid.write('%f\t%f\t%f\n'%(info[0], info[1], info[2]))
        else:
            fid.write('%f\t%f\t%f\n'%(0,0,0))
    count=count+1

fid.close()

