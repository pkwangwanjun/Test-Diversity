#!/usr/bin/env python2
# -*- coding: utf-8 -*-


from keras.preprocessing.image import ImageDataGenerator
from keras.applications import inception_v3
from keras.applications import inception_resnet_v2
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Conv2D, BatchNormalization,AveragePooling2D,Input
from keras.models import Model,load_model
from keras.regularizers import l2


from keras.applications import imagenet_utils

from keras.applications.inception_v3 import preprocess_input



import glob
import keras
import cv2
import os
import shutil
import numpy as np

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def mvdir():
    for dir_lst in glob.glob('/home/qingkaishi/imagenet/tiny-imagenet-200/train/*'):
        if len(glob.glob(dir_lst+'/*.txt'))!=0:
            os.remove(glob.glob(dir_lst+'/*.txt')[0])
        os.removedirs(dir_lst+'/images')
        for img in glob.glob(dir_lst+'/images/*'):
            shutil.move(img,dir_lst+'/'+os.path.basename(img))




class config():
    train_path='/home/qingkaishi/imagenet/tiny-imagenet-200/train'
    test_path='/mnt/datasets/test'
    batch_size=64
    num_sample=0
    for dir_path in glob.glob(train_path+'/*'):
        num_sample+=len(glob.glob(dir_path+'/*'))

    num_class=len(glob.glob(train_path+'/*'))
    input_shape=(64,64,3)
    target_size=(64,64)


def scale_data(img,full_shape=config.input_shape):
    (fh,fw,_)=full_shape
    #img=(255-img)
    #img=(img-128)/128.
    #img=cv2.resize(img,(full_shape[:2]))
    img=img.astype('float32')
    img/=255.
    #return img.reshape(fh,fw,3)
    return img


#导入数据集
#/home/qingkaishi/imagenet/tiny-imagenet-200/train
#/home/qingkaishi/imagenet/tiny-imagenet-200/val/images
#/home/qingkaishi/imagenet/tiny-imagenet-200/val/val_annotations.txt

#模型的构造
#inceptionV3
class Inceptionv3(object):
    @staticmethod
    def train():
        img=ImageDataGenerator(rotation_range=10,width_shift_range=0.05,height_shift_range=0.05,preprocessing_function=scale_data)
        train_generator=img.flow_from_directory(config.train_path,config.target_size,batch_size=config.batch_size,color_mode='rgb')
        base_model = inception_v3.InceptionV3(include_top=False, weights=None, input_shape=config.input_shape, classes=config.num_class)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(config.num_class, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        #opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        model_checkpoint=ModelCheckpoint('/home/qingkaishi/Test-Diversity/model_imagenet_tiny/model_v3.hdf5',monitor='loss',verbose=1,save_best_only=True)

        model.fit_generator(generator=train_generator,steps_per_epoch=config.num_sample//config.batch_size,
                            epochs=200,verbose=1,callbacks=[model_checkpoint],
                            )



class Resnet(object):
    @staticmethod
    def train():
        img=ImageDataGenerator(rotation_range=10,width_shift_range=0.05,height_shift_range=0.05,rescale=1/255)
        train_generator=img.flow_from_directory(config.train_path,config.target_size,batch_size=config.batch_size,color_mode='rgb')
        n=3
        depth = n * 6 + 2
        model = resnet_v1(input_shape=config.input_shape, depth=depth,num_classes=config.num_class)
        #opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        model_checkpoint=ModelCheckpoint('/home/qingkaishi/Test-Diversity/model_imagenet_tiny/model_v3.hdf5',monitor='loss',verbose=1,save_best_only=True)

        model.fit_generator(generator=train_generator,steps_per_epoch=config.num_sample//config.batch_size,
                            epochs=200,verbose=1,callbacks=[model_checkpoint],
                            )

#最小139*139
model=inception_v3.InceptionV3(include_top=True,weights='imagenet',input_tensor=None,input_shape=None,pooling=None,classes=1000)

img_lst=[]
for img in glob.glob('/home/qingkaishi/imagenet/tiny-imagenet-200/train/n03617480/*'):
    img_lst.append(cv2.cvtColor(cv2.resize(cv2.imread(img),(299,299))),cv2.COLOR_BGR2RGB)
img_lst=np.array(img_lst)

img_lst=imagenet_utils.preprocess_input(img_lst,mode='tf')
pred=model.predict(img_lst)
pred_top5=imagenet_utils.decode_predictions(pred,top=5)



#数据生成器
#img=ImageDataGenerator(rotation_range=10,width_shift_range=0.05,height_shift_range=0.05,preprocessing_function=scale_data)
#img=ImageDataGenerator(preprocessing_function=scale_data)
#train_generator=img.flow_from_directory(config.train_path,config.target_size,batch_size=config.batch_size,color_mode="grayscale")



if __name__=='__main__':
    pass
