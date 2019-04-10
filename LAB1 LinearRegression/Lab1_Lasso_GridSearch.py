#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 19:42:32 2019

@author: e1ixir
"""


import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression,Ridge,Lasso,\
LogisticRegression,ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing,\
fetch_20newsgroups_vectorized
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def evalue(model,y_test,x_test):
    y_predict=model.predict(x_test)
    mse=mean_squared_error(y_test,y_predict)
    mae=mean_absolute_error(y_test,y_predict)
    score=model.score(x_test,y_test)
    r2score=r2_score(y_test,y_predict)
    print("mse:{} \t mae:{}\t score:{}\t r2_score:{}\n"\
          .format(mse,mae,score,r2score))
    
# =============================================================================
#  导入房价数据
# =============================================================================
data=fetch_california_housing()
data_x=data.data
data_y=data.target
feature=data.feature_names

# =============================================================================
# 分割训练数据集
# =============================================================================
x_train,x_test,y_train,y_test=train_test_split(data_x,data_y,test_size=0.2,random_state=1)



# =============================================================================
# 参数调优
# =============================================================================

from sklearn.model_selection import GridSearchCV
parameter={"alpha":[0.1,0.5,1.0,0.05],"max_iter":[100,500,1000,50]}
#损失函数中的alpha 选择和最大迭代次数
clf=GridSearchCV(Lasso(),parameter,cv=5)
#5折交叉验证

# =============================================================================
# 训练模型（
# =============================================================================

clf.fit(x_train,y_train)

# =============================================================================
# 打印训练测试结果
# =============================================================================

# 自动参数调优结果
print("best params:{}".format(clf.best_params_))
#测试结果
evalue(clf,y_test,x_test)