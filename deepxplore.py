#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import keras
from keras import Model,Input
from keras.models import load_model
from diversity import generate_sample
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score



'''
0
109.45410614013679
425.4984741210939
1261.7449218750003
'''


'''
0
109.45410614013679
425.4984741210939
1261.7449218750003
'''


def neuron_coverge(image,layer1,layer2,layer3,layer4,threshold1,threshold2,threshold3,threshold4):


    layer1_act=((layer1.predict(image)>threshold1).sum(axis=0)>0).sum()
    layer1_num=26*26*16

    layer2_act=((layer2.predict(image)>threshold2).sum(axis=0)>0).sum()
    layer2_num=11*11*32

    layer3_act=((layer3.predict(image)>threshold3).sum(axis=0)>0).sum()
    layer3_num=3*3*32

    layer4_act=((layer4.predict(image)>threshold4).sum(axis=0)>0).sum()
    layer4_num=1*1*32

    ratio=(layer1_act+layer2_act+layer3_act+layer4_act)/float((layer1_num+layer2_num+layer3_num+layer4_num))

    return ratio


def Exp2(image,label):
    model=load_model('model/model.hdf5')
    pred=model.predict(image)

    acc=[]
    auc_macro=[]
    auc_micro=[]
    deepxplore=[]
    model=load_model('./model/model.hdf5')
    layer1=Model(inputs=model.layers[0].output,outputs=model.layers[2].output)
    layer2=Model(inputs=model.layers[0].output,outputs=model.layers[5].output)
    layer3=Model(inputs=model.layers[0].output,outputs=model.layers[8].output)
    layer4=Model(inputs=model.layers[0].output,outputs=model.layers[10].output)

    threshold1=np.percentile(layer1.predict(image).reshape(-1),90)
    threshold2=np.percentile(layer2.predict(image).reshape(-1),90)
    threshold3=np.percentile(layer3.predict(image).reshape(-1),90)
    threshold4=np.percentile(layer4.predict(image).reshape(-1),90)
    for i in tqdm(range(500)):

        index_choice=np.random.choice(range(image.shape[0]),size=600,replace=False)
        acc.append(accuracy_score(label[index_choice],np.argmax(pred[index_choice],axis=1)))
        auc_macro.append(roc_auc_score(pd.get_dummies(label[index_choice]),pred[index_choice],average='macro'))
        auc_micro.append(roc_auc_score(pd.get_dummies(label[index_choice]),pred[index_choice],average='micro'))
        ratio=neuron_coverge(image[index_choice],layer1,layer2,layer3,layer4,threshold1,threshold2,threshold3,threshold4)
        deepxplore.append(ratio)
        print(ratio)
    return acc,auc_macro,auc_micro,deepxplore







if __name__=='__main__':
    image=[]
    label=[]
    for i in range(10):
        org,fat,thin,adv=generate_sample(label=i,ratio=0.1)
        temp_image=np.concatenate([org,fat,thin,adv],axis=0)
        temp_label=i*np.ones(len(temp_image))
        image.append(temp_image.copy())
        label.append(temp_label.copy())
    image=np.concatenate(image,axis=0)
    label=np.concatenate(label,axis=0)
    acc,auc_macro,auc_micro,deepxplore=Exp2(image,label)
    df=pd.DataFrame([acc,auc_macro,auc_micro,deepxplore])
    df=df.T
