import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import nibabel as nib
import imageio
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
import gudhi as gd
import PIL.Image as Image
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from pylab import *
from sklearn.neural_network import MLPClassifier
import copy
from scipy import stats
dirlst=os.listdir(r'D:\Reslice_reho\Reslice_reho')
df=pd.read_excel(r'abide2_information.xlsx')
data=df[['subname','AGE_AT_SCAN ','SEX','DX_GROUP']]

print(data['DX_GROUP'])
feature_matrix=np.zeros((795,8))
feature_matrix=feature_matrix.tolist()
feature_matrix1=np.loadtxt('abide2residual_matrix_with_age')
feature_matrix1=feature_matrix1.tolist()
feature_lst=[21,26,27,28,29,30,36,57]
for i in range(795):
    for j in feature_lst:
        feature_matrix[i][feature_lst.index(j)]=feature_matrix1[i][j]
#print(feature_matrix)

import random
number=list(range(795))
random.shuffle(number)
for i in range(795):
    feature_matrix[i]=feature_matrix[number[i]]
    data.loc[i,'DX_GROUP']=data['DX_GROUP'][number[i]]
data,target=feature_matrix[:] ,data['DX_GROUP'][:]
for i in range(len(data)):
    for j in range(len(data[0])):
        data[i][j]=round(data[i][j],2)
data=np.array(data)
kf=KFold(n_splits=10,shuffle=True,random_state=0)
curr_score=0
spe_score=0
sen_score=0
f1_score=0
a=0

for train_index,test_index in kf.split(data):
    a+=1
    clt1 = GradientBoostingClassifier(max_depth=8,n_estimators=500,learning_rate=0.01, random_state=0).fit(data[train_index], target[train_index])
    curr_score = curr_score + clt1.score(data[test_index], target[test_index])
    print('准确率为：',clt1.score(data[test_index],target[test_index]))
    # print(clt1.predict_proba(data[test_index]))
    # print(data[test_index])
    #ro_curve(predict_exchange, target_exchange,r'{0}-Fold'.format(i))
    tp,fp,fn,tn=0,0,0,0
    for i in range(len(list(data[test_index]))):
        if list(clt1.predict(list(data[test_index])))[i]==1:
            if list(target[test_index])[i]==1:
                tp+=1
            else:
                fp+=1
        else:
            if list(target[test_index])[i]==1:
                fn += 1
            else:
                tn += 1
    tnr= tn/(fp+tn+0.001)
    tpr=tp/(tp+fn+0.001)
    recall = tp / (tp + fn + 0.001)
    precision = tp / (tp + fp)
    F1score = (2 * recall * precision) / (recall + precision)
    spe_score += tnr
    sen_score += tpr
    f1_score += F1score
    print('specificity为：', tnr)
    print('sensitivity为：', tpr)
    print('F1-score为:', F1score)

avg_score1=curr_score/10
avg_spe1=spe_score/10
avg_sen1=sen_score/10
avg_f1score4=f1_score/10
print('平均准确率为：',avg_score1)
print('平均specificity为：',avg_spe1)
print('平均sensitivity为:',avg_sen1)
print('平均F1score为：',avg_f1score4)


