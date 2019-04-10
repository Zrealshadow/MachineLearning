#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 19:39:08 2019

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
# 定义模型
# =============================================================================

#lr=LinearRegression()
l_ridge=Ridge(alpha=1.0)
#l_lasso=Lasso(alpha=0.1)
#l_elas=ElasticNet(random_state=0)

# =============================================================================
# 训练模型（只进行了一次迭代）
# =============================================================================

#lr.fit(x_train,y_train)
l_ridge.fit(x_train,y_train)
#l_lasso.fit(x_train,y_train)
#l_elas.fit(x_train,y_train)

# =============================================================================
# 打印训练测试结果
# =============================================================================

#evalue(lr,y_test,x_test)
evalue(l_ridge,y_test,x_test)
#evalue(l_lasso,y_test,x_test)
#evalue(l_elas,y_test,x_test)