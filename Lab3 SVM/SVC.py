#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:54:56 2019

@author: e1ixir
"""

# =============================================================================
#SVM 利用SVC对wine数据集进行分类
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
# 读取数据，PCA降维可视化
# =============================================================================

data=load_wine()
data_x=data.data
data_y=data.target
data_y=data_y.astype(np.int32)
feature=data.feature_names

#降维后可视化
pca=PCA(n_components=2)
x_pca=pca.fit_transform(data_x)
c1=x_pca[data_y==0]
c2=x_pca[data_y==1]
c3=x_pca[data_y==2]
plt.scatter(c1[:,0],c1[:,1],c='r',s=15,edgecolors='k')
plt.scatter(c2[:,0],c2[:,1],c='g',s=15,edgecolors='k')
plt.scatter(c3[:,0],c3[:,1],c='b',s=15,edgecolors='k')
plt.title('unpreprocessed data visiulization')
plt.legend(data.target_names)
plt.savefig('SVC_noprocessed_visulization.png')
plt.show()


#标准化后可视化
pca=PCA(n_components=2)
standard_x=StandardScaler().fit_transform(data_x)
x_pca=pca.fit_transform(standard_x)
c1=x_pca[data_y==0]
c2=x_pca[data_y==1]
c3=x_pca[data_y==2]
plt.scatter(c1[:,0],c1[:,1],c='r',s=15,edgecolors='k')
plt.scatter(c2[:,0],c2[:,1],c='g',s=15,edgecolors='k')
plt.scatter(c3[:,0],c3[:,1],c='b',s=15,edgecolors='k')
plt.title('Standard data visiulization')
plt.legend(data.target_names)
plt.savefig('SVC_Standard_visulization.png')
plt.show()


#归一化后可视化
pca=PCA(n_components=2)
Minmax=MinMaxScaler().fit_transform(data_x)
x_pca=pca.fit_transform(Minmax)
c1=x_pca[data_y==0]
c2=x_pca[data_y==1]
c3=x_pca[data_y==2]
plt.scatter(c1[:,0],c1[:,1],c='r',s=15,edgecolors='k')
plt.scatter(c2[:,0],c2[:,1],c='g',s=15,edgecolors='k')
plt.scatter(c3[:,0],c3[:,1],c='b',s=15,edgecolors='k')
plt.title('MaxMinScaler data visiulization')
plt.legend(data.target_names)
plt.savefig('SVC_Maxminscaler_visulization.png')
plt.show()


#PCA降维后 两个组分和原来特征之间的强弱关系
m=np.abs(pca.components_)
plt.matshow(m,cmap='plasma')
plt.yticks([0,1],['component1','component2'])
plt.colorbar()
plt.xticks(range(len(feature)),feature,rotation=60)
plt.savefig('SVC_PAC_relationship.png')
plt.show()


x_train,x_test,y_train,y_test=train_test_split(data_x,data_y,test_size=0.2,random_state=1)

#标准化
ss=StandardScaler().fit(x_train)
x_train_trans=ss.transform(x_train)
x_test_trans=ss.transform(x_test)

# =============================================================================
# 建模评分
# =============================================================================

#未标准化模型评分
svc=SVC()
svc.fit(x_train,y_train)
print('no preprocessed data \n the model score:{}\n'.format(svc.score(x_test,y_test)))


#标准化后的数据训练 模型评分

svc.fit(x_train_trans,y_train)

print('Standard data \n the model score:{}\n'.format(svc.score(x_test_trans,y_test)))







