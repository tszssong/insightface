import glob
import numpy
import os
SDPath =  '/data03/zhengmeisong/TestData/JA-Test/'
FDSets = ['JA-Test-Data/','xxx']

fid = open('../imgs.lst','w')
count=0

imgSet = FDSets[0]
GatRoots = glob.glob(SDPath+imgSet+'/*')
for idx2,imgID in enumerate(GatRoots):
    if idx2%1000==0:
        print(idx2)
    imgs=glob.glob(imgID+'/*.jpg')
    for imgName in imgs:
        fid.write(imgName+' %d\n'%count)
    count=count+1

fid.close()

