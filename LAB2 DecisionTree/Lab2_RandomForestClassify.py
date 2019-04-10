#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:54:06 2019

@author: e1ixir
"""

# =============================================================================
# 随机森林分类
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

#参数选择 为是 c4.5决策树还是CART决策树，以及最大深度，和最大利用特征数
parameter={"criterion":['gini','entropy'],"max_depth":[3,6,None],\
           "max_features":["auto","sqrt","log2",None]}
# 这里选择的超参数只选择了树的深度和使用信息增益度或gini指数作为分类依据
rfc=GridSearchCV(RandomForestClassifier(),parameter,cv=4)
rfc.fit(x_train,y_train)
evalue(rfc,y_test,x_test)
print("best params:{}".format(rfc.best_params_))





