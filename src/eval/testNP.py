import os, sys
import numpy as np
import time
EPS = 2000
def mergeSort(s, num_split = 40):
    ms = np.empty(s.shape[0])
    len_split = s.shape[0]//num_split
    s = np.append(s,EPS)  #扩展一个值，避免越界
    count = np.empty(num_split, dtype=int)
    baseSub = 0
    for idx in range(num_split):
        count[idx] = idx*len_split
        baseSub += idx*len_split
#     count = np.array([0,len_split, 2*len_split, 3*len_split])
#     baseSub = len_split + 2*len_split + 3*len_split
    print(count, baseSub)
    start = time.time()
    _done = False
    
#     while(count[0]<len_split or count[1]<2*len_split or count[2]<3*len_split or count[3]<4*len_split):
    while( not _done):
        tmp = s[count]
#         print(count, tmp)
        for idx in range(num_split):   #已达边界，不再参与比较
            if count[idx] == (idx+1)*len_split:
                tmp[idx] = EPS
        min_idx = np.argmin(tmp)
#         print(np.sum(count) - baseSub, tmp[min_idx])
        ms[np.sum(count) - baseSub] = tmp[min_idx]
        count[min_idx] += 1
        if count[0] % 1000000 == 0 and (time.time()-start) > 0.1:
            print(count, time.time()-start)
            start = time.time()
        for idx in range(num_split):
            if count[idx] == (idx+1)*len_split:
                _done = True
            else:
                _done = False
                continue
    return ms
def sepSort(a, num_split = 40, max_num = 1e8):
#     max_num = 50000*50000
    if a.shape[0] <= max_num:
        return(np.sort(a))
    else:
#     if True:
        print("Sort an array of %d seperate with %d"%(a.shape[0], num_split))
         
        len_split = a.shape[0]//num_split
        assert a.shape[0] % num_split == 0
        s = np.empty(a.shape[0])
        for idx in range(num_split):
            aa = a[idx*len_split:(idx+1)*len_split]   #.copy()
#             print(idx, aa.shape, aa)
            start = time.time()
            s[idx*len_split:(idx+1)*len_split] = np.sort(aa)
            print("%d sort %d use: %.2f s"%(idx, aa.shape[0], time.time()-start))
        start = time.time()
        s = mergeSort(s)
        print("merge array of %d use %.2f s"%(s.shape[0], time.time()-start))
        return s

if __name__ == "__main__":
    num_test = 20000
    num_sqrt = num_test*num_test
    print("test numpy sort")
    start = time.time()
    a = np.random.rand((num_sqrt))
    print("gen %dw**2 used %.2f s"%(int(num_test/10000), time.time()-start))
    start = time.time()
#     s = np.sort(a)
    s = sepSort(a)
    print("sort %dw**2 used %.2f s"%(int(num_test/10000), time.time()-start))
    print("before:", a)
    print("after :", s)
#     print("before:", a[:5])
#     print("after :", s[:5])
    
