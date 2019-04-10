#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:48:09 2019

@author: e1ixir
"""

# =============================================================================
# 随机森林回归
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

data_california_housing=fetch_california_housing()
#print(data_california_housing.DESCR)
data=data_california_housing.data
target=data_california_housing.target
features=data_california_housing.feature_names

x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=1)

# =============================================================================
# 所有特征进行评估，并进行网格自动调参数
# =============================================================================

#参数选择，因硬件原因，只选择调整最大深度，和选择特征标准
##调参过程可能回耗费时间
parameter_randomforest={"criterion":["mse","mae"],\
#                         "max_features":["auto","sqrt","log2"],\
#                         "max_depth":[3,5,7]
}

rfr=GridSearchCV(RandomForestRegressor(),parameter_randomforest,cv=3)
rfr.fit(x_train,y_train)
evalue(rfr,y_test,x_test)
print("best params:{}".format(rfr.best_params_))





