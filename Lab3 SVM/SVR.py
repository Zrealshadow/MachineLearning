#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:42:48 2019

@author: e1ixir
"""

# =============================================================================
#  SVM 回归分析 california_housing 数据集
# =============================================================================

from sklearn.svm import SVR,SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.datasets import load_wine,fetch_california_housing
from sklearn.metrics import accuracy_score,r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import  PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 读取，划分，预处理，数据集
# =============================================================================
data=fetch_california_housing()
data_x=data.data
data_y=data.target
features=data.feature_names
x_train,x_test,y_train,y_test=train_test_split(data_x,data_y,test_size=0.2,random_state=1)


#对数据集进行标准化
ss=StandardScaler().fit(x_train)
x_train_trans=ss.transform(x_train)
x_test_trans=ss.transform(x_test)


# =============================================================================
#  建模，评分
# =============================================================================

#用未预处理过的数据进行训练并评分
svr=SVR()
svr.fit(x_train,y_train)
print('no preprocessed data \n the model score:{}\n'.format(svr.score(x_test,y_test)))

#标准化后的数据进行训练并评分
svr=SVR()
svr.fit(x_train_trans,y_train)
print('Standarded data \n the model score:{}\n'.format(svr.score(x_test_trans,y_test)))

#增加 惩罚相C进行调参

parameter={'C':[5,10,30,50]}
model=GridSearchCV(SVR(),parameter,cv=3)
model.fit(x_train_trans,y_train)
print('the best parameter:{}'.format(model.best_params_))
print('the model score:{}\n'.format(svr.score(x_test_trans,y_test)))
