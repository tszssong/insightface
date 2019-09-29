import numpy as np
import os,sys,shutil
import string
import math
import time
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='face model evaluate')
parser.add_argument('--ftname', default='.ft', help='name of feat.')
parser.add_argument('--ftsize', default=128, type=int)
parser.add_argument('--imgRoot', help = 'imgRoot')
parser.add_argument('--listFile', help = 'imgFile')
parser.add_argument('--ftRoot', help = 'featRoot')
parser.add_argument('--model', default= 'None', help='model path')
parser.add_argument('--saveFP', type = int, default = 0)
parser.add_argument('--saveRoot', type = str, default = './')
FARs=[1e-2,1e-3,1e-4]
args = parser.parse_args()

def loadAllFeature2(imgListFile,ftExt = '.arc'):
    imgList = open(imgListFile, 'r').readlines()
    fullFeat = np.zeros([len(imgList), args.ftsize],dtype = np.float32)
    # fullFeat = np.zeros([len(imgList), 128],dtype = np.float32)
    ftDir = None
    isGal = np.zeros([len(imgList),1],dtype=np.int)
    if not args.model =='None':
        # ftDir = args.model + '/CASIA-IvS-Test/'  #TODO 
        ftDir = args.model + '/' + args.imgRoot.split('/')[-2] + '/'
    count = 0  #incase
    for idx,line in enumerate(imgList):
        ftName =line.split(' ')[0][:-4]+ftExt
        ft_name = ftName.split('/')[-1]
        if ft_name.find('_')<0:
            isGal[idx,0]=-1
        if not ftDir==None:
            ftElems = ftName.split('/')
            ftName = ftDir + ftElems[-2]+'/'+ftElems[-1]
        if idx%500==0:
            print('load ft %d. '%idx), 
            sys.stdout.flush()
        fullFeat[idx-count,:] = np.loadtxt(ftName)[np.newaxis,:]
    fullFeat = fullFeat[0:len(imgList)-count]
    print(count, len(imgList), fullFeat.shape)
    return fullFeat,isGal

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
    ey = np.ones([label.shape[0],label.shape[0]]) -2*np.eye(label.shape[0])
    for i in range(np.max(label)+1):
        idxs=np.where(label==i)[0]
        x[idxs[0]:idxs[-1]+1,idxs[0]:idxs[-1]+1]=np.ones([idxs.shape[0],idxs.shape[0]])
   
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
    # labelMatrix=getLabelMatrix_with_id_probe(testDataInfo['labels'],isGal)
    labelMatrix=getLabelMatrix(testDataInfo['labels'])
    labelArray = labelMatrix.reshape([1,testDataInfo['labels'].shape[0]**2])
    GenuNum = np.where(labelArray == 1)[0].shape[0]
    ImpoNum = np.where(labelArray == 0)[0].shape[0]
    print("GenuNum:%d, ImpoNum:%d"%(GenuNum, ImpoNum))
    nonSelfIdx = np.where(labelArray >-1 )[1]
   
    scores = (np.tensordot(testDataFeature, testDataFeature.transpose(), axes=1)+1)/2
    scoreArray = scores.reshape([1,testDataInfo['labels'].shape[0]**2])

    fLabels = labelArray[0,nonSelfIdx]
    fScores = scoreArray[0,nonSelfIdx]
    
    minScore = np.min(fScores)
    maxScore = np.max(fScores)
    print("maxScore:%.4f, minScore:%.4f"%(maxScore, minScore))
    # stepThr = max((maxScore - minScore) / 1000, 0.0001)
    stepThr = max((maxScore - minScore) / 10000, 0.00001)
    return fScores,fLabels,minScore,maxScore,stepThr,GenuNum,ImpoNum, nonSelfIdx,scores

def getRocCurve(fScores,fLabels,minScore,maxScore,stepThr,GenuNum,ImpoNum):
    ther = minScore
    TPRList = []
    FARList = []
    AccList = []
    PairNum=fScores.shape[0]
    ThrList = []
    while(ther <= maxScore):
        CurrTPNum = len(np.where(fScores*fLabels>=ther)[0])
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
    return FARArry, TPRArry, AccArry, ThrArry

def getRocCurveV2(fScores,fLabels,minScore,maxScore,stepThr,GenuNum,ImpoNum):
    ther = minScore
    TPRList = []
    FARList = []
    AccList = []
    PairNum=fScores.shape[0]
    ThrList = []
    negLabels = np.where(fLabels==0)[0]
    nfScores = fScores[negLabels]
    fsort = np.argsort(-nfScores)

    for idx,far in enumerate(FARs):
        negNums = int(round(ImpoNum*far))
        if negNums == 0:
            negNums = 1

        fsortx = fsort[:negNums]
        ther = nfScores[fsortx[-1]]

        CurrTPNum = len(np.where(fScores*fLabels>=ther)[0])
        CurrFPNum = len(np.where(fScores * (fLabels-1)*(-1) >= ther)[0])

        CurrTPR = float(CurrTPNum)/GenuNum
        CurrFAR = float(CurrFPNum)/ImpoNum

        CurrAcc = np.float( CurrTPNum + ( ImpoNum-CurrFPNum ) ) / PairNum
        TPRList.append(CurrTPR)
        FARList.append(CurrFAR)
        AccList.append(CurrAcc)
        ThrList.append(ther.copy())
        
    FARArry = np.asarray(FARList)
    TPRArry = np.asarray(TPRList)
    AccArry = np.asarray(AccList)
    ThrArry = np.asarray(ThrList)
        
    return FARArry, TPRArry, AccArry, ThrArry

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
    for idx,far in enumerate(FARs):
        print('%.9f(FPR)\t(%.9f(FPR))\t@\t%f(TPR)\t%f(Acc)\twith\t%f(Thr)'%(far,rFARs[idx],TPRs[idx],ACCs[idx],Thrs[idx]))
    return Thrs

# imgListPath = args.imgRoot+args.listFile
imgListPath = '/data03/zhengmeisong/wkspace/FR/doorbell/logs/ai_all_smallface_picked.lst'
ftName = args.ftname #'.dl23f1-80' #'.q2-146'
fScores,fLabels,minScore,maxScore,stepThr,GenuNum,ImpoNum,nonSelfIdx,scores=evaluateAllData(imgListPath,base_dir='',ftExt=ftName)#'.r50_grn3ft_tripletnd')#r50_grn1ft_color')
print(fScores.shape)
# for i in range(fScores.shape[0]):
#     print(fScores[i]," "),
minScore = np.max([minScore, 0.5])
maxScore = np.min([maxScore,0.9])
FARArry, TPRArry,AccArry,ThrArry=getRocCurveV2(fScores,fLabels,minScore,maxScore,0.0001,GenuNum,ImpoNum)
THRS=getFarValues(FARs,FARArry,TPRArry,AccArry,ThrArry)

scoreArray = scores.reshape([1, scores.shape[0]**2])
imgLit = open(imgListPath).readlines()

if False:
    for idx, thr in enumerate(THRS):
        
        fpairs  = getFalsePositives(fScores, fLabels,thr, nonSelfIdx)
        fnpairs = getFalseNegatives(fScores, fLabels,thr, nonSelfIdx)
        print(thr,'fpairs',len(fpairs),'npairs', len(fnpairs))
        if idx == 1 or idx == 2:
            continue
        fscores = scoreArray[0,fpairs][:,np.newaxis]
        fnscores = scoreArray[0,fnpairs][:,np.newaxis]

        vInfo   = np.concatenate([fpairs[:,np.newaxis],fscores],axis=1)
        saveDir = args.saveRoot + '/' + args.ftname
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)
        if not os.path.exists(saveDir+"/fp_imgs/"):
            os.mkdir(saveDir+"/fp_imgs/")
        if not os.path.exists(saveDir+"/fn_imgs/"):
            os.mkdir(saveDir+"/fn_imgs/")
        fName = saveDir+'/fpIdx_%d.txt'%idx
        fName2 = saveDir+'/fnIdx_%d.txt'%idx
        fid = open(fName,'w')
        fid2 = open(fName2,'w')
        for idx,ret in enumerate(zip(fpairs,fscores)):
            pairIdx , score = ret
            idA = pairIdx%scores.shape[0]
            idB = pairIdx/scores.shape[0]
            imgA = imgLit[idA].split(' ')[0]
            imgB = imgLit[idB].split(' ')[0]
            fid.write("%d\t%6.5f\t%s\t%s\n"%(pairIdx,score,imgA,imgB))
            shutil.copy(imgA,  saveDir+'/fp_imgs/'+str(pairIdx)+'_a.jpg')
            shutil.copy(imgB,  saveDir+'/fp_imgs/'+str(pairIdx)+'_b.jpg')
        for idx,ret in enumerate(zip(fnpairs,fnscores)):
            pairIdx , score = ret
            idA = pairIdx%scores.shape[0]
            idB = pairIdx/scores.shape[0]
            imgA = imgLit[idA].split(' ')[0]
            imgB = imgLit[idB].split(' ')[0]
            fid2.write("%d\t%6.5f\t%s\t%s\n"%(pairIdx,score,imgA,imgB))
            shutil.copy(imgA,  saveDir+'/fn_imgs/'+str(pairIdx)+'_a.jpg')
            shutil.copy(imgB,  saveDir+'/fn_imgs/'+str(pairIdx)+'_b.jpg')
    
        fid.close()
        fid2.close()
