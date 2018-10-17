#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras.applications import inception_v3
from keras.applications import imagenet_utils
from tensorflow.keras import Model,Input
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation,Flatten
import tensorflow as tf
import math
import numpy as np
import pandas as pd
import foolbox
import cv2
from tqdm import tqdm
import glob

import scipy

import networkx as nx

from scipy.linalg.misc import norm


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



def generate_sample():
    img_arr=[]
    for index in range(10000):
        path='/home/qingkaishi/imagenet/tiny-imagenet-200/val/images/val_{}.JPEG'.format(index)
        img=cv2.imread(path)
        img_arr.append(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    label=pd.read_csv('/home/qingkaishi/imagenet/tiny-imagenet-200/val/val_annotations.txt',sep='\t',header=None)
    return np.array(img_arr),label[1].values



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

def top5_score(label,pred_top5):
    return np.array([label[index] in label_one for index,label_one in enumerate(pred_top5[:,:,0])]).sum()

def Exp2(image,label):
    tf.keras.backend.set_learning_phase(0)

    model=inception_v3.InceptionV3(include_top=True,weights='imagenet',input_tensor=None,input_shape=None,pooling=None,classes=1000)

    image_arr=[]
    print('preprocess image')
    for img in tqdm(image):
        image_arr.append(cv2.resize(img,(299,299)))
    image_arr=np.array(image_arr)
    image_arr=imagenet_utils.preprocess_input(image_arr,mode='tf')
    #image_arr=image_arr[:4000]
    print('model predict')
    pred=model.predict(image_arr)
    pred_top5=imagenet_utils.decode_predictions(pred,top=5)
    pred_top5=np.array(pred_top5)

    acc=[]
    entropy=[]
    random_entropy=[]

    print('random select')

    for i in tqdm(range(1000)):
        #image.shape[0]
        index_choice=np.random.choice(range(image.shape[0]),size=2500,replace=False)
        temp1=top5_score(label[index_choice],pred_top5[index_choice])
        temp2=graph_distance(pred[index_choice])
        temp3=graph_distance(pred[index_choice],span=False)
        acc.append(temp1)
        entropy.append(temp2)
        random_entropy.append(temp3)

    return acc,entropy,random_entropy





if __name__=='__main__':
    image,label=generate_sample()
    acc,entropy,random_entropy=Exp2(image,label)
    df=pd.DataFrame([acc,entropy,random_entropy])
    df=df.T
    df.to_csv('output.csv')
