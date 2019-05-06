
# 加入角度信息的InsightFace

用insightface实现商汤[DREAM](https://github.com/penincillin/DREAM/)  

## 数据处理  
x1.用mxnet_tools/im2rec_1root2label.py 生成lst  --这个list和insightFace用的List多有不同，生成的rec id是错的  
1.用getRecImg/genLists/getImgList.py生成lst，如果有2个路径用getImgList2.py，多个需另写  
2.用src/data/face2rec_angle.py 生成带角度信息的rec文件  
3.用getRecImg/ang_img_iter.py验证生成的rec文件，打印信息里的id2range比真实id少一个（从0开始），根据id解小图，会将label和角度直接打印在图上  
-生成rec时会读取图片同名的info并写入pitch yaw roll三个角度，如果info为空，写入三个0  
-prefix.lst路径下放置property文件，num_classes,img_width,imgheight,其中num_classes无所谓没有用到
###### 生成好lst以后执行：
    python face2rec_angle.py prefix.lst  --num-thread 28
###### list格式如下，中间以tab作为分隔符  
![](deploy/lst格式.png)  

## 训练  
