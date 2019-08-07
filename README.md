

## 下载数据集
如果不使用Google Colab运行，需要自己下载数据集。
VAP RGBD face database： http://www.vap.aau.dk/rgb-d-face-database/
后5个解压到/faceid_val，剩下的解压到/faceid_train，注释faceID_train.py第一部分

## 训练
运行`python faceID_train.py`,训练的模型会保存到`faceid_big_rgbd.h5`
也可直接下载模型，链接：https://pan.baidu.com/s/1wPkrwyw1lO_kvpMC4GnmDw  ,提取码：aewz

## 运行
运行`python faceID_test.py`，返回两组rgbd图像之间的距离dis

## 测试结果
<img src='https://raw.githubusercontent.com/zj19941113/faceID_remake/master/img/1.png' width='800px'/>

<div style='text-align:center'>与同一用户不同姿势间距离</div>

<img src='https://raw.githubusercontent.com/zj19941113/faceID_remake/master/img/2.png' width='800px'/>

<div style='text-align:center'>与不同用户正脸间距离</div>

参考： https://towardsdatascience.com/how-i-implemented-iphone-xs-faceid-using-deep-learning-in-python-d5dbaa128e1d



