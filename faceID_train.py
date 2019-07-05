#pip install keras==2.1.5
import keras

# 一、下载数据集

# 数据集地址：https://translate.googleusercontent.com/translate_c?depth=1&rurl=translate.google.com.hk&sl=auto&sp=nmt4&tl=zh-CN&u=http://www.vap.aau.dk/rgb-d-face-database/&xid=17259,15700023,15700186,15700190,15700256,15700259&usg=ALkJrhgFnt5MU34ED2irHe9E7Zto24Lzbw
# 如果不是用Google Colab运行的，建议直接点开数据集地址，一个个下载后再解压到相对应的faceid_train或faceid_val，跳过 “一、下载数据集”部分

import os

os.mkdir("faceid_train")
os.mkdir("faceid_val")

link_list=["http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-16)(151751).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-16)(153054).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-16)(154211).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-16)(160440).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-16)(160931).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-16)(161342).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-16)(163349).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-16)(164248).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-17)(141550).zip", \
          "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-17)(142154).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-17)(142457).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-17)(143016).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(132824).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(133201).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(133846).zip", \
          "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(134239).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(134757).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(140516).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(143345).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(144316).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(145150).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(145623).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(150303).zip", \
          "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(150650).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(151337).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(151650).zip"]
val_list=["http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(152717).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(153532).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(154129).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(154728).zip", "http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(155357).zip"]

import requests, zipfile, io
for link in link_list:
  r = requests.get(link, stream=True)
  z = zipfile.ZipFile(io.BytesIO(r.content))
  z.extractall("faceid_train")
for link in val_list:
  r = requests.get(link, stream=True)
  z = zipfile.ZipFile(io.BytesIO(r.content))
  z.extractall("faceid_val")


# 二、数据集处理

import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image

def create_couple(file_path):
    folder = np.random.choice(glob.glob(file_path + "*"))
    while folder == "datalab":
        folder = np.random.choice(glob.glob(file_path + "*"))
    #  print(folder)
    mat = np.zeros((480, 640), dtype='float32')
    i = 0
    j = 0
    depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
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
    mat_small = (mat_small - np.mean(mat_small)) / np.max(mat_small)
    #    plt.imshow(mat_small)
    #    plt.show()

    mat2 = np.zeros((480, 640), dtype='float32')
    i = 0
    j = 0
    depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200 or int(val) == -1: val = 1200
                mat2[i][j] = float(int(val))
                j += 1
                j = j % 640

            i += 1
        mat2 = np.asarray(mat2)
    mat2_small = mat2[140:340, 220:420]
    mat2_small = (mat2_small - np.mean(mat2_small)) / np.max(mat2_small)
    #    plt.imshow(mat2_small)
    #    plt.show()
    return np.array([mat_small, mat2_small])

def create_couple_rgbd(file_path):
    folder = np.random.choice(glob.glob(file_path + "*"))
    while folder == "datalab":
        folder = np.random.choice(glob.glob(file_path + "*"))
    #  print(folder)
    mat = np.zeros((480, 640), dtype='float32')
    i = 0
    j = 0
    depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
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
    #    plt.imshow(mat_small)
    #    plt.show()
    #    plt.imshow(img)
    #    plt.show()

    mat2 = np.zeros((480, 640), dtype='float32')
    i = 0
    j = 0
    depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200 or int(val) == -1: val = 1200
                mat2[i][j] = float(int(val))
                j += 1
                j = j % 640

            i += 1
        mat2 = np.asarray(mat2)
    mat2_small = mat2[140:340, 220:420]
    img2 = Image.open(depth_file[:-5] + "c.bmp")
    img2.thumbnail((640, 480))
    img2 = np.asarray(img2)
    img2 = img2[160:360, 240:440]

    #   plt.imshow(img2)
    #   plt.show()
    mat2_small = (mat2_small - np.mean(mat2_small)) / np.max(mat2_small)
    #   plt.imshow(mat2_small)
    #   plt.show()

    full1 = np.zeros((200, 200, 4))
    full1[:, :, :3] = img[:, :, :3]
    full1[:, :, 3] = mat_small

    full2 = np.zeros((200, 200, 4))
    full2[:, :, :3] = img2[:, :, :3]
    full2[:, :, 3] = mat2_small
    return np.array([full1, full2])

def create_wrong(file_path):
    folder = np.random.choice(glob.glob(file_path + "*"))
    while folder == "datalab":
        folder = np.random.choice(glob.glob(file_path + "*"))
    mat = np.zeros((480, 640), dtype='float32')
    i = 0
    j = 0
    depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
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
    mat_small = (mat_small - np.mean(mat_small)) / np.max(mat_small)
    #   plt.imshow(mat_small)
    #   plt.show()

    folder2 = np.random.choice(glob.glob(file_path + "*"))
    while folder == folder2 or folder2 == "datalab":  # it activates if it chose the same folder
        folder2 = np.random.choice(glob.glob(file_path + "*"))
    mat2 = np.zeros((480, 640), dtype='float32')
    i = 0
    j = 0
    depth_file = np.random.choice(glob.glob(folder2 + "/*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200 or int(val) == -1: val = 1200
                mat2[i][j] = float(int(val))
                j += 1
                j = j % 640

            i += 1
        mat2 = np.asarray(mat2)
    mat2_small = mat2[140:340, 220:420]
    mat2_small = (mat2_small - np.mean(mat2_small)) / np.max(mat2_small)
    #   plt.imshow(mat2_small)
    #   plt.show()

    return np.array([mat_small, mat2_small])

def create_wrong_rgbd(file_path):
    folder = np.random.choice(glob.glob(file_path + "*"))
    while folder == "datalab":
        folder = np.random.choice(glob.glob(file_path + "*"))
    mat = np.zeros((480, 640), dtype='float32')
    i = 0
    j = 0
    depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
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
    #  plt.imshow(img)
    #  plt.show()
    #  plt.imshow(mat_small)
    #  plt.show()
    folder2 = np.random.choice(glob.glob(file_path + "*"))
    while folder == folder2 or folder2 == "datalab":  # it activates if it chose the same folder
        folder2 = np.random.choice(glob.glob(file_path + "*"))
    mat2 = np.zeros((480, 640), dtype='float32')
    i = 0
    j = 0
    depth_file = np.random.choice(glob.glob(folder2 + "/*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200 or int(val) == -1: val = 1200
                mat2[i][j] = float(int(val))
                j += 1
                j = j % 640

            i += 1
        mat2 = np.asarray(mat2)
    mat2_small = mat2[140:340, 220:420]
    img2 = Image.open(depth_file[:-5] + "c.bmp")
    img2.thumbnail((640, 480))
    img2 = np.asarray(img2)
    img2 = img2[140:340, 220:420]
    mat2_small = (mat2_small - np.mean(mat2_small)) / np.max(mat2_small)
    #   plt.imshow(img2)
    #   plt.show()
    #   plt.imshow(mat2_small)
    #   plt.show()
    full1 = np.zeros((200, 200, 4))
    full1[:, :, :3] = img[:, :, :3]
    full1[:, :, 3] = mat_small

    full2 = np.zeros((200, 200, 4))
    full2[:, :, :3] = img2[:, :, :3]
    full2[:, :, 3] = mat2_small
    return np.array([full1, full2])

print(create_couple("faceid_train/"))
create_couple_rgbd("faceid_val/")
create_wrong("faceid_train/")
create_wrong_rgbd("faceid_val/")[0].shape


# 三、网络结构

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU, concatenate, GlobalAveragePooling2D, Input, BatchNormalization, SeparableConv2D, Subtract, concatenate
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras import backend as K


def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))

# 定义损函
def contrastive_loss(y_true, y_pred):
    margin = 1.
    return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))
# return K.mean( K.square(y_pred) )

def fire(x, squeeze=16, expand=64):
    x = Convolution2D(squeeze, (1, 1), padding='valid')(x)
    x = Activation('relu')(x)

    left = Convolution2D(expand, (1, 1), padding='valid')(x)
    left = Activation('relu')(left)

    right = Convolution2D(expand, (3, 3), padding='same')(x)
    right = Activation('relu')(right)

    x = concatenate([left, right], axis=3)
    return x


img_input=Input(shape=(200,200,4))

x = Convolution2D(64, (5, 5), strides=(2, 2), padding='valid')(img_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

x = fire(x, squeeze=16, expand=16)
x = fire(x, squeeze=16, expand=16)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

x = fire(x, squeeze=32, expand=32)
x = fire(x, squeeze=32, expand=32)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

x = fire(x, squeeze=48, expand=48)
x = fire(x, squeeze=48, expand=48)
x = fire(x, squeeze=64, expand=64)
x = fire(x, squeeze=64, expand=64)
x = Dropout(0.2)(x)
x = Convolution2D(512, (1, 1), padding='same')(x)
out = Activation('relu')(x)

modelsqueeze= Model(img_input, out)
modelsqueeze.summary()


im_in = Input(shape=(200,200,4))
#wrong = Input(shape=(200,200,3))

x1 = modelsqueeze(im_in)
#x = Convolution2D(64, (5, 5), padding='valid', strides =(2,2))(x)
#x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x1)

"""
x1 = Convolution2D(256, (3,3), padding='valid', activation="relu")(x1)
x1 = Dropout(0.4)(x1)

x1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x1)

x1 = Convolution2D(256, (3,3), padding='valid', activation="relu")(x1)
x1 = BatchNormalization()(x1)
x1 = Dropout(0.4)(x1)

x1 = Convolution2D(64, (1,1), padding='same', activation="relu")(x1)
x1 = BatchNormalization()(x1)
x1 = Dropout(0.4)(x1)
"""

x1 = Flatten()(x1)

x1 = Dense(512, activation="relu")(x1)
x1 = Dropout(0.2)(x1)
#x1 = BatchNormalization()(x1)
feat_x = Dense(128, activation="linear")(x1)
feat_x = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_x)

model_top = Model(inputs = [im_in], outputs = feat_x)
model_top.summary()

im_in1 = Input(shape=(200,200,4))
im_in2 = Input(shape=(200,200,4))

feat_x1 = model_top(im_in1)
feat_x2 = model_top(im_in2)

lambda_merge = Lambda(euclidean_distance)([feat_x1, feat_x2])

model_final = Model(inputs = [im_in1, im_in2], outputs = lambda_merge)
model_final.summary()

adam = Adam(lr=0.001)
sgd = SGD(lr=0.001, momentum=0.9)
model_final.compile(optimizer=adam, loss=contrastive_loss)


# 四、训练参数

def generator(batch_size):
    while 1:
        X = []
        y = []
        switch = True
        for _ in range(batch_size):
            #   switch += 1
            if switch:
                #   print("correct")
                X.append(create_couple_rgbd("faceid_train/").reshape((2, 200, 200, 4)))
                y.append(np.array([0.]))
            else:
                #   print("wrong")
                X.append(create_wrong_rgbd("faceid_train/").reshape((2, 200, 200, 4)))
                y.append(np.array([1.]))
            switch = not switch
        X = np.asarray(X)
        y = np.asarray(y)
        XX1 = X[0, :]
        XX2 = X[1, :]
        yield [X[:, 0], X[:, 1]], y

def val_generator(batch_size):
    while 1:
        X = []
        y = []
        switch = True
        for _ in range(batch_size):
            if switch:
                X.append(create_couple_rgbd("faceid_val/").reshape((2, 200, 200, 4)))
                y.append(np.array([0.]))
            else:
                X.append(create_wrong_rgbd("faceid_val/").reshape((2, 200, 200, 4)))
                y.append(np.array([1.]))
            switch = not switch
        X = np.asarray(X)
        y = np.asarray(y)
        XX1 = X[0, :]
        XX2 = X[1, :]
        yield [X[:, 0], X[:, 1]], y

gen = generator(16)
val_gen = val_generator(4)

outputs = model_final.fit_generator(gen, steps_per_epoch=30, epochs=50, validation_data = val_gen, validation_steps=20)


# 五、保存模型

model_final.save("faceid_big_rgbd.h5")

# 六、测试模型

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
    plt.figure(figsize=(8, 8))
    plt.grid(True)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(mat_small)
    plt.show()
    plt.figure(figsize=(8, 8))
    plt.grid(True)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.show()

    full1 = np.zeros((200, 200, 4))
    full1[:, :, :3] = img[:, :, :3]
    full1[:, :, 3] = mat_small

    return np.array([full1])

file1 = ('faceid_val/(2012-05-18)(153532)/005_2_d.dat')
inp1 = create_input_rgbd(file1)
file1 = ('faceid_val/(2012-05-18)(153532)/001_1_d.dat')
inp2 = create_input_rgbd(file1)

model_final.predict([inp1, inp2])