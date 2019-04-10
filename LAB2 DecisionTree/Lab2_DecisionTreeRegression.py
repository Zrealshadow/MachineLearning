#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:37:42 2019

@author: e1ixir
"""

# =============================================================================
# 决策树回归分析，2个特征及多个特征
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

# =============================================================================
# 随机选择两个特征，并可视化
# =============================================================================

show=pd.DataFrame(data,columns=features)
l=[0,1,2,3,4,5,6,7]
k=random.sample(l,2)
data_random_feature=show.iloc[:,k]
x_train,x_test,y_train,y_test=train_test_split(data_random_feature,target,test_size=0.2,random_state=1)

#决策树回归建模并评估
regression_tree=DecisionTreeRegressor(max_depth=3,random_state=1)
regression_tree.fit(x_train,y_train)
print("2 feature selection:\n")
evalue(regression_tree,y_test,x_test)


#决策树可视化
dot_data=export_graphviz(regression_tree,out_file=None,\
                        feature_names=[features[k[0]],features[k[1]]],rounded=True,filled=True)
graph=pydotplus.graph_from_dot_data(dot_data)
graph.write_png('./regressiontreeOf2Features.png')


# =============================================================================
# 所有特征进行评估，并进行网格自动调参数
# =============================================================================
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=1)
#参数选择，因硬件原因，只选择调整最大深度，和选择特征标准
##调参过程可能回耗费时间
parameter_regression={"criterion":["mse","mae"],\
#                       "splitter":["best","random"],\
#                       "max_features":["auto","sqrt","log2"],\
#                      "max_depth":[3,5,7]
}

dtr=GridSearchCV(DecisionTreeRegressor(),parameter_regression,cv=3)
dtr.fit(x_train,y_train)
print("all feature selection:\n")
evalue(dtr,y_test,x_test)
print("best params:{}".format(dtr.best_params_))





