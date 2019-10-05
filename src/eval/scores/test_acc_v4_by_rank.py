# author: i-xiaoshengtao@360.cn
# this script loads features of images from image_list with(image_name, id_label,pts,pts,pts)
# performs pair-wise verification process
# outputs: TPRs at different FARs with Thresholds

import sys
import numpy as np
import os
import string
import math
import time
import argparse

parser = argparse.ArgumentParser(description='face model evaluate')
parser.add_argument('--ftname', default='.ft', help='name of feat.')
parser.add_argument('--imgRoot', help = 'imgRoot')
parser.add_argument('--listFile', help = 'imgFile')
parser.add_argument('--ftRoot', help = 'featRoot')
parser.add_argument('--model', default= 'None', help='model path')
parser.add_argument('--saveFP', type = int, default = 1)
FARs=[1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]

args = parser.parse_args()

def loadAllFeature3(imgListFile,ftExt = '.arc'):
    imgList = open(imgListFile, 'r').readlines()
    fullFeat = np.zeros([len(imgList), 512],dtype = np.float32)
    for idx,line in enumerate(imgList):
        if idx%1000==0:
            print('load ft %d'%idx),
            sys.stdout.flush()
        imgName = line.split(' ')[0]
        ftName = imgName[:-4]+ftExt
        ft = np.loadtxt(ftName)
        fid = open(ftName);
        ft =  np.array([line.strip().split() for line in fid],float)[:,0]
        fullFeat[idx,:] = ft[np.newaxis,:]
    return fullFeat


def loadAllFeature2(imgListFile,ftExt = '.arc'):
    imgList = open(imgListFile, 'r').readlines()
    fullFeat = np.zeros([len(imgList), 512],dtype = np.float32)
    ftDir = None
    isGal = np.zeros([len(imgList),1],dtype=np.int)
    if not args.model =='None':
        ftDir = args.model.split(',')[0]+'-%04d/'%int(args.model.split(',')[-1])
    for idx,line in enumerate(imgList):
        ftName =line.split(' ')[0][:-4]+ftExt
        if ftName.find('id')>0:
            isGal[idx,0]=-1
        if not ftDir==None:
            ftElems = ftName.split('/')
            ftName = ftDir + ftElems[-2]+'/'+ftElems[-1]
        if idx%1000==0:
            print('load ft %d'%idx),
            sys.stdout.flush()
        fullFeat[idx,:] = np.loadtxt(ftName)[np.newaxis,:]
    return fullFeat,isGal


def loadAllFeature(imgListFile,ftExt = '.arc'):
    imgList = open(imgListFile, 'r').readlines()
    for idx,line in enumerate(imgList):
        if idx%500==0:
            print('load ft %d'%idx)
        imgName = line.split(' ')[0]
        ftName = imgName[:-4]+ftExt
        ft = np.loadtxt(ftName)
        if idx == 0:
            fullFeat = ft[np.newaxis,:]
        else:
            fullFeat = np.concatenate([fullFeat,ft[np.newaxis,:]],axis=0)
    return fullFeat

def getDataStatsOfPairs(imgListFile,base_dir='',ftExt='.arc'):
    imgList = open(imgListFile,'r').readlines()
    testData={}
    testData['labels']=[]
    testData['imgNames']=[]
    for idx,line in enumerate(imgList):
        imgName = line.split(' ')[0]
        label = int(line.split(' ')[1])
        testData['imgNames'].append(imgName)
        if idx == 0:
            testData['labels'] = np.array([label]).reshape([1,1])
        else:
            testData['labels'] = np.concatenate([testData['labels'],np.array([label]).reshape([1,1])],axis=0)
    return testData

def getLabelMatrix_with_id_probe(label,isGal):
    x = np.zeros([label.shape[0],label.shape[0]])
    ey =np.ones([label.shape[0],label.shape[0]]) -2*np.eye(label.shape[0])
    for i in range(np.max(label)+1):
        idxs=np.where(label==i)[0]
        x[idxs[0]:idxs[-1]+1,idxs[0]:idxs[-1]+1]=np.ones([idxs.shape[0],idxs.shape[0]])
    #x=np.multiply(x,ey)
    labelInfo=np.dot(isGal, isGal.T)
    isProbe = np.abs(np.abs(isGal)-1)
    labelProbeInfo=np.dot(isProbe,isProbe.T)

    x=np.multiply(x,ey)
    x = x - labelInfo
    y = np.logical_and(x, labelProbeInfo)
    x = x - labelProbeInfo
    
    x = x-2*y
    return x

def getLabelMatrix(label):
    x = np.zeros([label.shape[0],label.shape[0]])
    ey =np.ones([label.shape[0],label.shape[0]]) -2*np.eye(label.shape[0])
    for i in range(np.max(label)+1):
        idxs=np.where(label==i)[0]
        x[idxs[0]:idxs[-1]+1,idxs[0]:idxs[-1]+1]=np.ones([idxs.shape[0],idxs.shape[0]])
    x=np.multiply(x,ey)
    return x

def evaluateAllData(imgListFile,base_dir='',ftExt='.arc'):
    testDataFeature,isGal = loadAllFeature2(imgListFile, ftExt=ftExt)
    testDataInfo = getDataStatsOfPairs(imgListFile, base_dir='', ftExt=ftExt)
    labelMatrix=getLabelMatrix_with_id_probe(testDataInfo['labels'],isGal)
    labelArray = labelMatrix.reshape([1,testDataInfo['labels'].shape[0]**2])
    GenuNum = np.where(labelArray == 1)[0].shape[0]
    ImpoNum = np.where(labelArray == 0)[0].shape[0]
    nonSelfIdx = np.where(labelArray >-1 )[1]
    print(np.min(labelArray[:,nonSelfIdx]))

    scores = (np.tensordot(testDataFeature, testDataFeature.transpose(), axes=1)+1)/2
    scoreArray = scores.reshape([1,testDataInfo['labels'].shape[0]**2])

    fLabels = labelArray[0,nonSelfIdx]
    fScores = scoreArray[0,nonSelfIdx]
    G1 = np.where(scoreArray > 1.0)[1]
    minScore = np.min(fScores)
    maxScore = np.max(fScores)
    print(maxScore, minScore)
    stepThr = max((maxScore - minScore) / 1000, 0.0001)
    return fScores,fLabels,minScore,maxScore,stepThr,GenuNum,ImpoNum, nonSelfIdx,scores

def getRocCurve(fScores,fLabels,minScore,maxScore,stepThr,GenuNum,ImpoNum):
    ther = minScore
    TPRList = []
    FARList = []
    AccList = []
    PairNum=fScores.shape[0]
    ThrList = []
    while(ther <= maxScore):

        CurrTPNum=len(np.where(fScores*fLabels>=ther)[0])
        CurrFPNum = len(np.where(fScores * (fLabels-1)*(-1) >= ther)[0])

        CurrTPR = float(CurrTPNum)/GenuNum
        CurrFAR = float(CurrFPNum)/ImpoNum

        CurrAcc = np.float( CurrTPNum + ( ImpoNum-CurrFPNum ) ) / PairNum
        TPRList.append(CurrTPR)
        FARList.append(CurrFAR)
        AccList.append(CurrAcc)
        ThrList.append(ther.copy())
        ther = ther + stepThr

    FARArry = np.asarray(FARList)
    TPRArry = np.asarray(TPRList)
    AccArry = np.asarray(AccList)
    ThrArry = np.asarray(ThrList)

    return FARArry, TPRArry,AccArry,ThrArry


def getRocCurveV2(fScores,fLabels,minScore,maxScore,stepThr,GenuNum,ImpoNum):
    ther = minScore
    TPRList = []
    FARList = []
    AccList = []
    PairNum=fScores.shape[0]
    ThrList = []
    #while(ther <= maxScore):
    negLabels = np.where(fLabels==0)[0]
    nfScores = fScores[negLabels]
    fsort = np.argsort(-nfScores)

    for idx,far in enumerate(FARs):

        negNums = int(round(ImpoNum*far))
        if negNums == 0:
            negNums = 1

        fsortx=fsort[:negNums]
        #if idx>3:
        #    print (nfScores[fsortx])
        ther = nfScores[fsortx[-1]]
        #print(ImpoNum,negNums,ther)

        CurrTPNum=len(np.where(fScores*fLabels>=ther)[0])
        CurrFPNum = len(np.where(fScores * (fLabels-1)*(-1) >= ther)[0])

        CurrTPR = float(CurrTPNum)/GenuNum
        CurrFAR = float(CurrFPNum)/ImpoNum

        CurrAcc = np.float( CurrTPNum + ( ImpoNum-CurrFPNum ) ) / PairNum
        TPRList.append(CurrTPR)
        FARList.append(CurrFAR)
        AccList.append(CurrAcc)
        ThrList.append(ther.copy())
        #ther = ther + stepThr

    FARArry = np.asarray(FARList)
    TPRArry = np.asarray(TPRList)
    AccArry = np.asarray(AccList)
    ThrArry = np.asarray(ThrList)

    #plt.plot(FARList,TPRList,'r')
    #plt.savefig('./roc.png')
    return FARArry, TPRArry,AccArry,ThrArry

def getFalsePositives(fScores, fLabels, thre, nonSelfIdx):
    FPidx= np.where((fScores * (fLabels-1)*(-1) >= thre)*(fLabels==0))[0]
    return nonSelfIdx[FPidx]

def getFalseNegatives(fScores, fLabels, thre, nonSelfIdx):
    FPidx= np.where( (fScores * fLabels < thre)*(fLabels==1) )[0]
    return nonSelfIdx[FPidx]

def getFarValues(FARs,FARArry,TPRArry,AccArry,ThrArry):
    FARArrys=np.repeat(FARArry[np.newaxis,:],len(FARs),axis=0)
    distances = np.abs(FARArrys-np.asarray(FARs)[:,np.newaxis])
    minIdxs = np.argmin(distances,axis=1)
    rFARs = FARArry[minIdxs]
    TPRs = TPRArry[minIdxs]
    Thrs = ThrArry[minIdxs]
    ACCs = AccArry[minIdxs]

    #print('TPR\t(cTPR)\t@\tFR\twith\tThr')
    print(args.model)
    for idx,far in enumerate(FARs):
        print('%.9f(FPR)\t(%.9f(FPR))\t@\t%f(TPR)\t%f(Acc)\twith\t%f(Thr)'%(far,rFARs[idx],TPRs[idx],ACCs[idx],Thrs[idx]))
    return Thrs

imgListPath = args.imgRoot+args.listFile
ftName = args.ftname #'.dl23f1-80' #'.q2-146'
fScores,fLabels,minScore,maxScore,stepThr,GenuNum,ImpoNum,nonSelfIdx,scores=evaluateAllData(imgListPath,base_dir='',ftExt=ftName)#'.r50_grn3ft_tripletnd')#r50_grn1ft_color')
minScore = np.max([minScore, 0.5])
maxScore = np.min([maxScore,0.9])
FARArry, TPRArry,AccArry,ThrArry=getRocCurveV2(fScores,fLabels,minScore,maxScore,0.0025,GenuNum,ImpoNum)
THRS=getFarValues(FARs,FARArry,TPRArry,AccArry,ThrArry)

scoreArray = scores.reshape([1, scores.shape[0]**2])
imgLit = open(imgListPath).readlines()
if args.saveFP==1:
    for idx, thr in enumerate(THRS):
        if idx ==0 or idx == 1:
            continue
        fpairs  = getFalsePositives(fScores, fLabels,thr, nonSelfIdx)
        fnpairs = getFalseNegatives(fScores, fLabels,thr, nonSelfIdx)
        print(thr,'fpairs',len(fpairs),'npairs', len(fnpairs))
        fscores = scoreArray[0,fpairs][:,np.newaxis]
        fnscores = scoreArray[0,fnpairs][:,np.newaxis]

        vInfo   = np.concatenate([fpairs[:,np.newaxis],fscores],axis=1)
    
        if not os.path.exists("./log/"):
            os.mkdir("./log/")
        fName = './log/fpIdx_%d.txt'%idx
        fName2 = './log/fnIdx_%d.txt'%idx
        fid = open(fName,'w')
        fid2 = open(fName2,'w')
        for idx,ret in enumerate(zip(fpairs,fscores)):
            pairIdx , score = ret
            idA = pairIdx%scores.shape[0]
            idB = pairIdx/scores.shape[0]
            imgA = imgLit[idA].split(' ')[0]
            imgB = imgLit[idB].split(' ')[0]
            fid.write("%d\t%6.5f\t%s\t%s\n"%(pairIdx,score,imgA,imgB))
        for idx,ret in enumerate(zip(fnpairs,fnscores)):
            pairIdx , score = ret;
            idA = pairIdx%scores.shape[0]
            idB = pairIdx/scores.shape[0]
            imgA = imgLit[idA].split(' ')[0]
            imgB = imgLit[idB].split(' ')[0]
            fid2.write("%d\t%6.5f\t%s\t%s\n"%(pairIdx,score,imgA,imgB))
    
        fid.close()
        fid2.close()
