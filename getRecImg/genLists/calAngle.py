import sys, os
import numpy as np
listpath = sys.argv[1]
widefile = open('./wide'+listpath, 'w')
widedir = open('./widedir'+listpath, 'w')
print(listpath)
imglist = open(listpath, 'r')
yaw_dict = {}
wide_file_list = []
count = 0
wide_cnt = 0
for line in imglist.readlines():
    items = line.strip().split('\t')
    yaw = abs(int(float(items[4]))) #down to int
    if not yaw in yaw_dict:
        yaw_dict[yaw] = []
    yaw_dict[yaw].append(items[1])
    if yaw > 40:
        wide_cnt += 1
        widefile.write(line)
        wide_path = items[1].split('/')[-2]
        if not wide_path in wide_file_list:
            wide_file_list.append(wide_path)
    count += 1
    if count % 1000 == 0:
        print count, "processed!"
print len(wide_file_list), "ids", wide_cnt, "faces has wide angle"
for label in wide_file_list:
    widedir.write(label+'\n')
widedir.close()
imglist.close()
widefile.close()
