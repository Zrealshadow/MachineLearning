#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 08:47:18 2019

@author: e1ixir
"""

# =============================================================================
# MLPClassify 
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
def evalue_classify(model,x,y):
    y_=model.predict(x)
    print("model info: \nactivation:{}\thidden_layer_size: {}".format(model.activation,model.hidden_layer_sizes))
    print("accuracy_score: {}".format(accuracy_score(y,y_)))
    print("layers: {}\titer: {}".format(model.n_layers_,model.n_iter_))
    print("the final loss:{}".format(model.loss_))
    print("*"*25+"\n")

#输入数据
    
data=load_wine()
data_x=data.data
data_y=data.target
feature=data.feature_names

#PCA降维，标准化数据，可视化
pca=PCA(n_components=2)
x_pca=pca.fit_transform(data_x)
train_x,test_x,train_y,test_y=train_test_split(x_pca,data_y,test_size=0.2,random_state=2)
ss=StandardScaler().fit(train_x)
trans_train_x=ss.transform(train_x)
trans_test_x=ss.transform(test_x)

model=MLPClassifier(activation="relu",hidden_layer_sizes=(32,16,8),random_state=2,max_iter=1000)
model.fit(trans_train_x,train_y)
evalue_classify(model,trans_test_x,test_y)

cmap_light=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
camp_bold=ListedColormap(['#FF0000','#00FF00','#0000FF'])
x_min,x_max=trans_train_x[:,0].min()-1,trans_train_x[:,0].max()+1
y_min,y_max=trans_train_x[:,1].min()-1,trans_train_x[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
z=model.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,z,cmap=cmap_light)
plt.scatter(trans_train_x[:,0],trans_train_x[:,1],c=train_y,edgecolor='k',s=60)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("MLPClassifier")
plt.show()


#评估所有特征
#预处理数据
x_train,x_test,y_train,y_test=train_test_split(data_x,data_y,test_size=0.2,random_state=2)
ss=StandardScaler().fit(x_train)
x_trans_train=ss.transform(x_train)
x_trans_test=ss.transform(x_test)


#调参数
model=MLPClassifier(activation="relu",hidden_layer_sizes=(32,16,8,4),random_state=2)
model.fit(x_trans_train,y_train)
evalue_classify(model,x_trans_test,y_test)

model=MLPClassifier(activation="logistic",hidden_layer_sizes=(32,16,8),random_state=2,max_iter=1000)
model.fit(x_trans_train,y_train)
evalue_classify(model,x_trans_test,y_test)

model=MLPClassifier(activation="logistic",hidden_layer_sizes=(32,16,8),random_state=2,max_iter=300)
model.fit(x_trans_train,y_train)
evalue_classify(model,x_trans_test,y_test)

model=MLPClassifier(activation="tanh",hidden_layer_sizes=(32,16,8),random_state=2,max_iter=200)
model.fit(x_trans_train,y_train)
evalue_classify(model,x_trans_test,y_test)


