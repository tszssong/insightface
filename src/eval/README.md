getFeatBatch.py 批量forward,batchsize=40 r100占单卡显存约6000MB  
testNP 为了解决np.sort数据量大时的内存占用问题，一个不成功的np.sort+merge函数，2万x2万的计算需要3小时，花了大约小半天，此事暂时搁置    