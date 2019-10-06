import glob
import numpy

import os
<<<<<<< HEAD
SDPath =  '/data3/zhengmeisong/data/gl2ms1m_dl23W1f1_150WW1_img/'
=======
SDPath =  '//data03/zhengmeisong/data/gl2ms1m_dl23W1f1_150WW1_img/'
>>>>>>> 5cc38bbd74d2da69c874a3df007cccb0111cc677
FDSets = ['imgs/','xxx']

fid = open('casia_jr.lst','w')
count=0

imgSet = FDSets[0]
GatRoots = glob.glob(SDPath+imgSet+'/*')
for idx2,imgID in enumerate(GatRoots):
<<<<<<< HEAD
    sub_path = imgID.split('/')[-1] 
    int_sub = int(sub_path.split('_')[-1])  
=======
>>>>>>> 5cc38bbd74d2da69c874a3df007cccb0111cc677
    if int_sub<63834274:
        continue
    if idx2%1000==0:
        print(idx2)
    imgs=glob.glob(imgID+'/*.jpg')
    for imgName in imgs:
        fid.write('1\t'+imgName+'\t%d\n'%count)
    count=count+1

<<<<<<< HEAD
SDPath =  '/data3/zhengmeisong/data/jrPairs/'
FDSets = ['imgs/','xxx']
=======
SDPath =  '/data03/zhengmeisong/jrData/'
FDSets = ['jr-pairs/','xxx']
>>>>>>> 5cc38bbd74d2da69c874a3df007cccb0111cc677

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

