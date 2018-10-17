#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
mnist模型训练程序
acc:0.9931
val_loss: 0.0332
val_acc: 0.9899
[0.03317314764151742, 0.9899]
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
from keras.datasets import mnist


def model_mnist():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()  # 28*28

    X_train = X_train.astype('float32').reshape(-1,28,28,1)
    X_test = X_test.astype('float32').reshape(-1,28,28,1)
    X_train /= 255
    X_test /= 255
    print('Train:{},Test:{}'.format(len(X_train),len(X_test)))

    nb_classes=10

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
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

    model.fit(X_train, Y_train, batch_size=64, nb_epoch=10,validation_data=(X_test, Y_test))
    model.save('./model/model_mnist.hdf5')
    score=model.evaluate(X_test, Y_test, verbose=0)
    print(score)

if __name__=='__main__':
    model_mnist()
