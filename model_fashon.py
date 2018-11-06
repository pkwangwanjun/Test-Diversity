#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
fashon_mnist模型训练程序
acc:0.9096
val_loss: 0.3039937790036201
val_acc: 0.8897
[0.3039937790036201, 0.8897]
'''

from keras import Model,Input
import sys
import time
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten,Input
from keras.models import Model,load_model
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import cv2




#导入数据集
import sys
sys.path.append('/home/qingkaishi/Diversity/Test-diversity/fashion-mnist/utils')
import mnist_reader




def model_fashion():
    path='/home/qingkaishi/Diversity/Test-diversity/fashion-mnist/data/fashion'
    X_train, y_train = mnist_reader.load_mnist(path, kind='train')
    X_test, y_test = mnist_reader.load_mnist(path, kind='t10k')

    X_train = X_train.astype('float32').reshape(-1,28,28,1)
    X_test = X_test.astype('float32').reshape(-1,28,28,1)
    X_train /= 255
    X_test /= 255
    print('Train:{},Test:{}'.format(len(X_train),len(X_test)))

    nb_classes=10

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print('data success')

    input_tensor=Input((28,28,1))
    #28*28
    temp=Conv2D(filters=16,kernel_size=(3,3),padding='valid',use_bias=False)(input_tensor)
    temp=Activation('relu')(temp)
    #26*26
    temp=MaxPooling2D(pool_size=(2, 2))(temp)
    #13*13
    temp=Conv2D(filters=32,kernel_size=(3,3),padding='valid',use_bias=False)(temp)
    temp=Activation('relu')(temp)
    #11*11
    temp=MaxPooling2D(pool_size=(2, 2))(temp)
    #5*5
    temp=Conv2D(filters=32,kernel_size=(3,3),padding='valid',use_bias=False)(temp)
    temp=Activation('relu')(temp)
    #3*3
    temp=Conv2D(filters=32,kernel_size=(3,3),padding='valid',use_bias=False)(temp)
    temp=Activation('relu')(temp)
    #1*1
    temp=Flatten()(temp)

    output=Dense(nb_classes,activation='softmax')(temp)

    model=Model(input=input_tensor,outputs=output)

    model.summary()

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=64, nb_epoch=10,validation_data=(X_test, y_test))
    model.save('./model/model_fashion.hdf5')
    score=model.evaluate(X_test, y_test, verbose=0)
    print(score)

if __name__=='__main__':
    model_fashion()
