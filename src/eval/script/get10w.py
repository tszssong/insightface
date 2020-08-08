import glob
import numpy
import os
SDPath =  '/ai_data/zhengmeisong/data/jaTrain/'
FDSets = ['imgs/','xxx']

fid = open('../imgs12wPairs.lst','w')
count=0

imgSet = FDSets[0]
GatRoots = glob.glob(SDPath+imgSet+'/*')
for idx2,imgID in enumerate(GatRoots):
    if idx2%1000==0:
        print(idx2)
    imgs=glob.glob(imgID+'/*.jpg')
    if len(imgs) < 2:
        continue
    _has_face = False
    _has_id   = False
    names = []
    for imgName in imgs:
      if 'face' in imgName:
        if not _has_face:
          names.append(imgName)
        _has_face = True
      if 'id' in imgName:
        if not _has_id:
          names.append(imgName)
        _has_id = True
    if _has_face and _has_id:
        for name in names:
            fid.write(name+' %d\n'%count)
        count=count+1
    

fid.close()

