import os, sys
import random
# def gen
if __name__ == "__main__":
    print("get pairs:")
    faces = {}
    with open('imgs2w.lst', 'r') as fr:
        lines = fr.readlines()
        print("total imgs:", len(lines))
        for idx, line in enumerate(lines):
          path, id = line.strip().split(' ')
          if id in faces:
              faces[id].append(path)
          else:
              faces[id] = [path]
    print("total ids:", len(faces))    

    with open('tmp.txt','a') as fw:
        fid_list = list(faces.keys())
        for fid, path in faces.items():
            if len(path)<2:
              print(fid,path)
              continue
            nid = fid
            while nid == fid:
              nid = random.sample(fid_list, k=1)[0]
            fw.write("%s\t%s\t%d\n"%(path[0], path[1], 1))  
            fw.write("%s\t%s\t%d\n"%(path[0], faces[nid][0], 0))  
