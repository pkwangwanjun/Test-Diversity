#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from tensorflow.keras import Model,Input
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation,Flatten
import tensorflow as tf
import math
import numpy as np
import pandas as pd
import foolbox
#from matplotlib import pyplot as plt
#import cv2
from tqdm import tqdm
#导入数据集
from tensorflow.keras.datasets import cifar10

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

import scipy

import networkx as nx

from scipy.linalg.misc import norm

import multiprocessing
import keras

import warnings

warnings.filterwarnings('ignore')


#diversity_vector=Model(inputs=model.layers[0].output,outputs=model.layers[-2].output)

#d_vector=diversity_vector.predict(X_test)


def count_entropy_layers(diversity_vector,x):
    def entropy(x):
        #epsilon=10e-8
        return sum(-x*np.log(x))
    d_vector=diversity_vector.predict(x)
    temp=d_vector
    temp=temp.sum(axis=0)/len(temp)
    score=entropy(temp)
    return score



def count_entropy(model,x):

    def entropy(x):
        #epsilon=10e-8
        return sum(-x*np.log(x))

    d_vector=model.predict(x)
    temp=d_vector
    temp=temp.sum(axis=0)/len(temp)
    score=entropy(temp)
    return score


def adv_example(x,y):
    keras.backend.set_learning_phase(0)
    model=load_model('saved_models/cifar10_ResNet20v1_model.125.h5')
    foolmodel=foolbox.models.KerasModel(model,bounds=(0,1),preprocessing=(0,1))

    attack=foolbox.attacks.IterativeGradientAttack(foolmodel)
    #attack=foolbox.attacks.DeepFoolL2Attack(foolmodel)
    result=[]
    for image in tqdm(x):
        #adv=attack(image.reshape(28,28,-1),label=y,steps=1000,subsample=10)
        adv=attack(image.reshape(32,32,-1),y,epsilons=[0.01,0.1],steps=100)
        if isinstance(adv,np.ndarray):
            result.append(adv)
        else:
            print('adv fail')
    return np.array(result)




def generate_sample(label,ratio=0.1):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 28*28
    X_train = X_train.astype('float32').reshape(-1,32,32,3)
    X_test = X_test.astype('float32').reshape(-1,32,32,3)
    X_train /= 255
    X_test /= 255

    Y_train=Y_train.reshape(-1)
    Y_test=Y_test.reshape(-1)

    image_org=X_test[Y_test==label]

    choice_index=np.random.choice(range(len(image_org)),size=int(len(image_org)*ratio),replace=False)
    image_org=image_org[choice_index]

    adv=adv_example(image_org,label)
    return image_org,adv



def graph(x,span=True):
    '''
    x是采样的图片向量
    '''
    num=len(x)
    G=nx.complete_graph(num)

    arr=np.ones((num,num))
    weight_edge=[]
    for i in range(num):
        for j in range(num):
            temp=norm(x[i]-x[j])
            arr[i][j]=temp
            weight_edge.append((i,j,temp))
            #weight_edge=[(i,j,norm(x[i]-x[j])) for i in range(num) for j in range(num)]
    G.add_weighted_edges_from(weight_edge)

    if span:
        span_tree=nx.minimum_spanning_tree(G)
        span_edge=span_tree.edges()
        #epsilons=10e-30
    else:
        choice_index=np.random.choice(range(len(G.edges())),size=num-1,replace=False)
        span_edge=np.array(G.edges())[choice_index]

    w_sum=0
    p=[]
    for index in span_edge:
        p.append(arr[tuple(index)])
        w_sum+=arr[tuple(index)]
    p=np.array(p)/w_sum
    result=-p*np.log(p)
    result[np.isnan(result)]=0
    return result.sum()


def graph_distance(x,span=True):
    '''
    x是采样的图片向量
    '''
    num=len(x)
    G=nx.complete_graph(num)

    arr=np.ones((num,num))
    weight_edge=[]
    for i in range(num):
        for j in range(num):
            temp=norm(x[i]-x[j])
            arr[i][j]=temp
            weight_edge.append((i,j,temp))
            #weight_edge=[(i,j,norm(x[i]-x[j])) for i in range(num) for j in range(num)]
    G.add_weighted_edges_from(weight_edge)
    if span:
        span_tree=nx.minimum_spanning_tree(G)
        span_edge=span_tree.edges()
        #epsilons=10e-30
    else:
        choice_index=np.random.choice(range(len(G.edges())),size=num-1,replace=False)
        span_edge=np.array(G.edges())[choice_index]
    w_sum=0
    p=[]
    for index in span_edge:
        p.append(arr[tuple(index)])
        w_sum+=arr[tuple(index)]
    p=np.array(p)/w_sum
    result=-p*np.log(p)
    result[np.isnan(result)]=0
    return w_sum*(result.sum())



def Exp(label):

    org,fat,thin,adv=generate_sample(label)

    img_concat=np.concatenate((org,fat,thin,adv),axis=0)
    num=len(img_concat)
    lst=[]

    model=load_model('model/model.hdf5')
    diversity_vector1=Model(inputs=model.layers[0].output,outputs=Activation('softmax',name='last')(Flatten()(model.layers[-4].output)))
    diversity_vector2=Model(inputs=model.layers[0].output,outputs=Activation('softmax',name='last2')(Flatten()(model.layers[-6].output)))
    diversity_vector3=Model(inputs=model.layers[0].output,outputs=Activation('softmax',name='last3')(Flatten()(model.layers[5].output)))

    for i in tqdm(range(1000)):
        index=np.random.choice(range(num),size=int(0.1*num),replace=False)
        temp=model.predict(img_concat[index])
        acc=accuracy_score(label*np.ones(int(num*0.1)),np.argmax(temp,axis=1))

        ce1=count_entropy(model,img_concat[index])
        ce2=count_entropy_layers(diversity_vector1,img_concat[index])
        ce3=count_entropy_layers(diversity_vector2,img_concat[index])
        ce4=count_entropy_layers(diversity_vector3,img_concat[index])

        lst.append([acc,ce1,ce2,ce3,ce4])
        return pd.DataFrame(lst,columns=['acc','ce1','ce2','ce3','ce4'])

def Exp2(image,label,size):
    tf.keras.backend.set_learning_phase(0)
    model=load_model('saved_models/cifar10_ResNet20v1_model.126.h5')
    pred=model.predict(image)

    #diversity_vector1=Model(inputs=model.layers[0].output,outputs=(model.layers[-2].output))

    #pred_last=diversity_vector1.predict(image)

    acc=[]
    entropy=[]
    random_entropy=[]
    last_entropy=[]
    auc_macro=[]
    auc_micro=[]
    for i in tqdm(range(500)):

        index_choice=np.random.choice(range(image.shape[0]),size=size,replace=False)

        acc.append(accuracy_score(label[index_choice],np.argmax(pred[index_choice],axis=1)))
        entropy.append(graph_distance(pred[index_choice]))
        random_entropy.append(graph_distance(pred[index_choice],span=False))
        #last_entropy.append(graph_distance(pred_last[index_choice]))
        #auc_macro.append(roc_auc_score(pd.get_dummies(label[index_choice]),pred[index_choice],average='macro'))
        #auc_micro.append(roc_auc_score(pd.get_dummies(label[index_choice]),pred[index_choice],average='micro'))

    return acc,entropy,random_entropy

def pool_func(size):
    '''
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_train = X_train.astype('float32').reshape(-1,32,32,3)
    X_test = X_test.astype('float32').reshape(-1,32,32,3)
    X_train /= 255
    X_test /= 255
    '''
    image=[]
    label=[]
    for i in range(10):
        org,fat=generate_sample(label=i,ratio=1)
        temp_image=np.concatenate([org,fat],axis=0)
        temp_label=i*np.ones(len(temp_image))
        image.append(temp_image.copy())
        label.append(temp_label.copy())
    image=np.concatenate(image,axis=0)
    label=np.concatenate(label,axis=0)
    acc,entropy,random_entropy=Exp2(image,label,size=size)
    df=pd.DataFrame([acc,entropy,random_entropy])
    df.to_csv('./label/output_all_class_cifar_{}.csv'.format(size))




if __name__=='__main__':
    pool_func(100)
    '''
    pool = multiprocessing.Pool(processes=6)
    for size in [50,100,200,500,800,1000]:
        pool.apply_async(pool_func, (size, ))
    pool.close()
    pool.join()
    '''
    '''
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_train = X_train.astype('float32').reshape(-1,32,32,3)
    X_test = X_test.astype('float32').reshape(-1,32,32,3)
    X_train /= 255
    X_test /= 255

    Y_train=Y_train.reshape(-1)
    Y_test=Y_test.reshape(-1)

    image=X_test
    label=Y_test

    acc,entropy,random_entropy,last_entropy,auc_macro,auc_micro=Exp2(image,label)
    df=pd.DataFrame([acc,entropy,random_entropy,last_entropy,auc_macro,auc_micro])
    df=df.T
    df.to_csv('output.csv')
    '''
lst=[]
lst_random=[]
for i in [50,100,200,500,800,1000]:
    path='output_all_class_cifar_{}.csv'.format(i)
    data=pd.read_csv(path,index_col=0).T
    lst.append(stats.pearsonr(data[0],data[1]))
    lst_random.append(stats.pearsonr(data[0],data[2]))
