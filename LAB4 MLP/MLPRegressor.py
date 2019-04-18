#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 08:36:23 2019

@author: e1ixir
"""

# =============================================================================
# 多层感知机回归 
# =============================================================================

from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,accuracy_score
from sklearn.datasets import  fetch_california_housing,load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#评估函数
def evalue(model,x,y):
    y_=model.predict(x)
    print("model info: \nactivation:{}\thidden_layer_size: {}".format(model.activation,model.hidden_layer_sizes))
    print("r2_score: {}\nmae: {}\nmsq: {}".format(r2_score(y,y_),mean_absolute_error(y,y_),mean_squared_error(y,y_)))
    print("layers: {}\titer: {}".format(model.n_layers_,model.n_iter_))
    print("the final loss:{}".format(model.loss_))
    print("*"*25+"\n")

#获取数据

data=fetch_california_housing()
data_x=data.data
data_y=data.target
features=data.feature_names
train_x,test_x,train_y,test_y=train_test_split(data_x,data_y,test_size=0.2,random_state=1)


#标准化数据
ss=StandardScaler()
ss.fit(train_x)
trans_train_x=ss.transform(train_x)
trans_test_x=ss.transform(test_x)


#未标准化数据模型评估
model=MLPRegressor(activation="logistic",hidden_layer_sizes=(36,18),random_state=1)
#激活函数sigmoid，两层隐藏层
model.fit(train_x,train_y)
print('No standard data training:')
evalue(model,test_x,test_y)


#调参 标准化数据
model=MLPRegressor(activation="logistic",hidden_layer_sizes=(32,16,8,4),random_state=1)
model.fit(trans_train_x,train_y)
evalue(model,trans_test_x,test_y)

model=MLPRegressor(activation="relu",hidden_layer_sizes=(32,16,8,4),random_state=1)
model.fit(trans_train_x,train_y)
evalue(model,trans_test_x,test_y)

model=MLPRegressor(activation="tanh",hidden_layer_sizes=(32,16,8,4),random_state=1)
model.fit(trans_train_x,train_y)
evalue(model,trans_test_x,test_y)

model=MLPRegressor(activation="tanh",hidden_layer_sizes=(64,32,16,8,4),random_state=1)
model.fit(trans_train_x,train_y)
evalue(model,trans_test_x,test_y)


