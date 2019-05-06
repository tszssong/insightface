
# 加入角度信息的InsightFace

用insightface实现商汤[DREAM](https://github.com/penincillin/DREAM/)  

## 数据处理  
1.用mxnet_tools/im2rec_1root2label.py 生成lst  
2.用src/data/face2rec_angle.py 生成带角度信息的rec文件  
-生成rec时会读取图片同名的info并写入pitch yaw roll三个角度，如果info为空，写入三个0  
-prefix.lst路径下放置property文件，num_classes,img_width,imgheight,其中num_classes无所谓没有用到
###### 执行：
    python im2rec_1root2label.py prefix root --list --reccursive
    python face2rec_angle.py prefix.lst  --num-thread 28
###### list格式如下，中间以tab作为分隔符  
![](deploy/lst格式.png)  

## 训练  
