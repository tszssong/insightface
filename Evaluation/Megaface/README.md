
Download megaface testpack data from [baiducloud](https://pan.baidu.com/s/1h4ezfwJiXClbZDdg1RX0MQ) or [dropbox](https://www.dropbox.com/s/5ko2dcd1x7vn37w/megaface_testpack_v1.0.zip?dl=0) and unzip it to ``data/``, then check and call ``run.sh`` to evaluate insightface model performance.

在insightface基础上做了少量修改，印象中megaface官方devkit在centos上没安装成功，由于1080ti机器没装gpu版mxnet，用cpu版提特征并计算，增加对pytorch模型的支持  
