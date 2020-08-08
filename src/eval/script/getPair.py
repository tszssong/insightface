import glob
import numpy
import os
SDPath =  '/cloud_data01_zzzc/zhengmeisong/data/gl2ms1m_imgs/'
FDSets = ['imgs/','xxx']

fid = open('../imgsPairs.lst','w')
count=0

imgSet = FDSets[0]
GatRoots = glob.glob(SDPath+imgSet+'/*')
for idx2,imgID in enumerate(GatRoots):
    if idx2%1000==0:
        print(idx2)
    imgs=glob.glob(imgID+'/*.jpg')
    if len(imgs) < 2:
        continue
    names = []
    for iidx,imgName in enumerate(imgs):
      if iidx>=2: 
        continue
      fid.write(imgName+' %d\n'%count)
    count=count+1

fid.close()

