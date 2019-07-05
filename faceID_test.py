# 一、加载模型

#pip install keras==2.1.5
from keras.models import load_model
from keras import backend as K

# 定义损函
def contrastive_loss(y_true, y_pred):
    margin = 1.
    return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))
# return K.mean( K.square(y_pred) )

model=load_model('faceid_big_rgbd.h5',custom_objects={'contrastive_loss':contrastive_loss})


# 二、模型测试

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def create_input_rgbd(file_path):
    #  print(folder)
    mat = np.zeros((480, 640), dtype='float32')
    i = 0
    j = 0
    depth_file = file_path
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200 or int(val) == -1: val = 1200
                mat[i][j] = float(int(val))
                j += 1
                j = j % 640

            i += 1
        mat = np.asarray(mat)
    mat_small = mat[140:340, 220:420]
    img = Image.open(depth_file[:-5] + "c.bmp")
    img.thumbnail((640, 480))
    img = np.asarray(img)
    img = img[140:340, 220:420]
    mat_small = (mat_small - np.mean(mat_small)) / np.max(mat_small)
    # plt.figure(figsize=(8, 8))
    # plt.grid(True)
    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(mat_small)
    # plt.show()
    # plt.figure(figsize=(8, 8))
    # plt.grid(True)
    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(img)
    # plt.show()

    full1 = np.zeros((200, 200, 4))
    full1[:, :, :3] = img[:, :, :3]
    full1[:, :, 3] = mat_small

    return np.array([full1])

file1 = ('faceid_val/(2012-05-18)(153532)/005_2_d.dat')
# file1 = ('faceid_val/(2012-05-18)(154728)/002_1_d.dat')
inp1 = create_input_rgbd(file1)
file1 = ('faceid_val/(2012-05-18)(153532)/001_1_d.dat')
# file1 = ('faceid_val/(2012-05-18)(155357)/013_2_d.dat')
inp2 = create_input_rgbd(file1)

print(model.predict([inp1, inp2])[0][0])

# 三、大规模测试
