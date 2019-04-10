#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:01:27 2019

@author: e1ixir
"""


# =============================================================================
# 决策树分类
# =============================================================================

from sklearn.datasets import fetch_california_housing, load_wine
from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score,\
mean_squared_error
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier,\
ExtraTreeClassifier,ExtraTreeRegressor,export_graphviz
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import numpy as np
import pandas as pd
import graphviz
import pydotplus
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# =============================================================================
# 评估函数
# =============================================================================

def evalue(model,y_test,x_test):
    y_predict=model.predict(x_test)
    mse=mean_squared_error(y_test,y_predict)
    mae=mean_absolute_error(y_test,y_predict)
    score=model.score(x_test,y_test)
    r2score=r2_score(y_test,y_predict)
    print("mse:{} \nmae:{}\nscore:{}\nr2_score:{}\n".format(mse,mae,score,r2score))
    
# =============================================================================
# 数据导入
# =============================================================================

data=load_wine()
x_data=data.data
y_data=data.target
features=data.feature_names
target_class=data.target_names
#print(target_class)
#print(features)
show=pd.DataFrame(x_data,columns=features)
#print(show.head())
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2,random_state=1)
# =============================================================================
# 所有特征进行评估，并进行网格自动调参数
# =============================================================================

parameter={"criterion":['gini','entropy'],"max_depth":[3,6,None],\
           "max_features":["auto","sqrt","log2",None]}
# 这里选择的超参数只选择了树的深度和使用信息增益度或gini指数作为分类依据
dtc=GridSearchCV(DecisionTreeClassifier(),parameter,cv=4)
dtc.fit(x_train,y_train)
evalue(dtc,y_test,x_test)
print("best params:{}".format(dtc.best_params_))


# =============================================================================
# 数据可视化
# =============================================================================


map_light=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
map_bold=ListedColormap(['#FF0000','#00FF00','#0000FF'])

wine=load_wine()
#取数据集中的前 2 个特征
x=wine.data[:,:2]
y=wine.target
#拆分数据集 
x_train,x_test,y_train,y_test=train_test_split(x,y)         
x_min,x_max=x_train[:,0].min()-1,x_train[:,0].max()+1 
y_min,y_max=x_train[:,1].min()-1,x_train[:,1].max()+1 
xx,yy=np.meshgrid(np.arange(x_min,x_max,.2),np.arange(y_min,y_max,.2)) 
dtf_vision=DecisionTreeClassifier()
dtf_vision.fit(x_train,y_train)
z=dtf_vision.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,z,cmap=map_light)
plt.scatter(x[:,0],x[:,1],c=y,cmap=map_bold,edgecolor='black',s=20)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.show()
                          
                          
                          
                          
                          
                          
                        
                        