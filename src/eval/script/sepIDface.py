import numpy
import os,sys
inlist = '../imgs1k.lst'
idlist = '../id1k.lst'
facelist = '../face1k.lst'
fid = open(idlist, 'w')
fface = open(facelist, 'w')
with open(inlist, 'r') as fp:
  lines = fp.readlines()
  for line in lines:
    if '_id' in line:
      fid.write(line)
    else:
      fface.write(line)
fid.close()
fface.close()

