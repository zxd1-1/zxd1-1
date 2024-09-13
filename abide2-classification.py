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

feature_matrix=np.loadtxt('abide2residual_matrix_with_age')
feature_matrix=feature_matrix.tolist()
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
fig=plt.figure(figsize=(8,8))
sub=fig.add_subplot(111)
sub.set_xticks([0.0,0.2,0.4,0.6,0.8,1.0])
sub.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
ax=gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlim((0.0,1.05))
plt.ylim((0.0,1.05))
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
feature_importance=list(np.zeros(73))
for train_index,test_index in kf.split(data):
    a+=1
    clt1 = GradientBoostingClassifier(max_depth=8,n_estimators=500,learning_rate=0.01, random_state=0).fit(data[train_index], target[train_index])
    curr_score = curr_score + clt1.score(data[test_index], target[test_index])
    print('准确率为：',clt1.score(data[test_index],target[test_index]))
    for i in range(73):
        feature_importance[i]+=clt1.feature_importances_[i]
    print(clt1.feature_importances_)
    target_exchange=[]
    predict_exchange=[]
    for item in target[test_index]:
        if item== 2:
            item=0
            target_exchange.append(item)
        else:
            target_exchange.append(item)
    for item in clt1.predict_proba(list(data[test_index])):
            predict_exchange.append(item[0])
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
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    F1score = (2 * recall * precision) / (recall + precision)
    spe_score += tnr
    sen_score += tpr
    f1_score += F1score
    print('specificity为：', tnr)
    print('sensitivity为：', tpr)
    print('F1-score为:', F1score)
    y_label = np.array(target_exchange)
    y_pred = np.array(predict_exchange)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(y_label, y_pred)
    roc_auc[0] = auc(fpr[0], tpr[0])

    sub.plot(fpr[0], tpr[0],
             lw=2, label=r'{0}-Fold'.format(a) + ' (area = %0.2f)' % roc_auc[0])
    sub.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
avg_score1=curr_score/10
avg_spe1=spe_score/10
avg_sen1=sen_score/10
avg_f1score1=f1_score/10
print('平均准确率为：',avg_score1)
print('平均specificity为：',avg_spe1)
print('平均sensitivity为:',avg_sen1)
print('平均F1score为：',avg_f1score1)
mean_feature_importance=list(np.zeros(73))
for i in range(73):
    mean_feature_importance[i]=feature_importance[i]/10
print(mean_feature_importance)
copy1=copy.deepcopy(mean_feature_importance)
copy1.sort()
d=copy1[-30:]
print(d)
for i in d:
    print(mean_feature_importance.index(i))
kf=KFold(n_splits=10,shuffle=True,random_state=0)
curr_score=0
spe_score=0
sen_score=0
f1_score=0
for train_index,test_index in kf.split(data):
    clt2 = RandomForestClassifier(max_depth=8,n_estimators=500, random_state=0).fit(data[train_index], target[train_index])
    curr_score = curr_score + clt2.score(data[test_index], target[test_index])
    print('准确率为：',clt2.score(data[test_index],target[test_index]))
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(list(data[test_index]))):
        if list(clt2.predict(list(data[test_index])))[i] == 1:
            if list(target[test_index])[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if list(target[test_index])[i] == 1:
                fn += 1
            else:
                tn += 1
    tnr = tn / (fp + tn+0.001)
    tpr = tp / (tp + fn+0.001)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    F1score = (2 * recall * precision) / (recall + precision)
    spe_score += tnr
    sen_score += tpr
    f1_score += F1score
    print('specificity为：', tnr)
    print('sensitivity为：', tpr)
    print('F1-score为:', F1score)
avg_score2=curr_score/10
avg_spe2=spe_score/10
avg_sen2=sen_score/10
avg_f1score2=f1_score/10
print('平均准确率为：',avg_score2)
print('平均specificity为：',avg_spe2)
print('平均sensitivity为:',avg_sen2)
print('平均F1score为：',avg_f1score2)


kf=KFold(n_splits=10,shuffle=True,random_state=0)
curr_score=0
spe_score=0
sen_score=0
f1_score=0
for train_index,test_index in kf.split(data):
    clf=MLPClassifier(solver='lbfgs', alpha=1e-5,activation = 'logistic',
                  hidden_layer_sizes=(50,50,20), random_state=0,max_iter=10000)
   # clt=DecisionTreeClassifier(max_depth=5,random_state=0).fit(data[train_index],target[train_index])
    clt3 = clf.fit(data[train_index], target[train_index])
    curr_score = curr_score + clt3.score(data[test_index], target[test_index])
    print('准确率为：',clt3.score(data[test_index],target[test_index]))
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(list(data[test_index]))):
       if list(clt3.predict(list(data[test_index])))[i] == 1:
          if list(target[test_index])[i] == 1:
              tp += 1
          else:
              fp += 1
       else:
          if list(target[test_index])[i] == 1:
              fn += 1
          else:
              tn += 1
    tnr = tn / (fp + tn+0.001)
    tpr = tp / (tp + fn+0.001)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    F1score = (2 * recall * precision) / (recall + precision)
    spe_score += tnr
    sen_score += tpr
    f1_score += F1score
    print('specificity为：', tnr)
    print('sensitivity为：', tpr)
    print('F1-score为:', F1score)

avg_score3=curr_score/10
avg_spe3=spe_score/10
avg_sen3=sen_score/10
avg_f1score3=f1_score/10
print('平均准确率为：',avg_score3)
print('平均specificity为：',avg_spe3)
print('平均sensitivity为:',avg_sen3)
print('平均F1score为：',avg_f1score3)

kf=KFold(n_splits=10,shuffle=True,random_state=0)
curr_score=0
spe_score=0
sen_score=0
f1_score=0
for train_index,test_index in kf.split(data):
    clf=svm.SVC(C=2,kernel='rbf')
   # clt=DecisionTreeClassifier(max_depth=5,random_state=0).fit(data[train_index],target[train_index])
    clt4 = clf.fit(data[train_index], target[train_index])
    curr_score = curr_score + clt4.score(data[test_index], target[test_index])
    print('准确率为：',clt4.score(data[test_index],target[test_index]))
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(list(data[test_index]))):
       if list(clt4.predict(list(data[test_index])))[i] == 1:
          if list(target[test_index])[i] == 1:
              tp += 1
          else:
              fp += 1
       else:
          if list(target[test_index])[i] == 1:
              fn += 1
          else:
              tn += 1
    tnr = tn / (fp + tn+0.001)
    tpr = tp / (tp + fn+0.001)
    recall = tp / (tp + fn+0.001)
    precision = tp / (tp + fp)
    F1score = (2 * recall * precision) / (recall + precision)
    spe_score += tnr
    sen_score += tpr
    f1_score += F1score
    print('specificity为：', tnr)
    print('sensitivity为：', tpr)
    print('F1-score为:', F1score)

avg_score4=curr_score/10
avg_spe4=spe_score/10
avg_sen4=sen_score/10
avg_f1score4=f1_score/10
print('平均准确率为：',avg_score4)
print('平均specificity为：',avg_spe4)
print('平均sensitivity为:',avg_sen4)
print('平均F1score为：',avg_f1score4)
