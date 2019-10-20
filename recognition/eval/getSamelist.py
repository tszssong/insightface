import sys, os
import shutil
def getPic(root, listname):
  pic_dict = {}
  with open(listname) as fp:
    lines = fp.readlines()
    for line in lines:
      fullpath = line.strip().split(' ')[0]
      filename = fullpath.split('/')[-1]
      sub_dir = fullpath.split('/')[-2]
      if not sub_dir in pic_dict:
        pic_dict[sub_dir] = [filename]
      else:
        pic_dict[sub_dir].append(filename)
  return pic_dict
      
def findPairs(root, pic_dict, numPairs=8000):
  posCount = 0
  assert numPairs%2==0
  negCount = numPairs/2
  if not os.path.isdir('./tmp'):
    os.makedirs('./tmp')
  with open('issame.lst', 'w') as fp:
    for key, value in pic_dict.iteritems():
      if(len(value)<2):
        continue
      isValid = isFace = isID = False
      for name in value:
        if '_face' in name: 
          isFace=True
          facename = name
        if '_id' in name: 
          isID = True
          idname = name
      isValid = isFace and isID
      if isValid:
        if posCount < (numPairs/2):
          shutil.copy(root+key+'/'+idname, './tmp/'+str(posCount)+'_1.jpg')
          shutil.copy(root+key+'/'+facename, './tmp/'+str(posCount)+'_2.jpg')
          fp.write(str(posCount)+' True\n')
          posCount += 1
        if posCount > 2:
          shutil.copy(root+key+'/'+idname, './tmp/'+str(negCount)+'_1.jpg')
          shutil.copy('./tmp/'+str(posCount-2)+'_2.jpg', './tmp/'+str(negCount)+'_2.jpg')
          fp.write(str(negCount)+' False\n')
          negCount += 1
      if negCount>=numPairs:
        break
   

if __name__ == '__main__':
  dataRoot = '/cloud_data01/zhengmeisong/wkspace/qh_recog_clean/deploy_st/tmp/'
  picDict = getPic(dataRoot, 'ja.lst')
  findPairs(dataRoot, picDict, 8000)
  print(picDict)

