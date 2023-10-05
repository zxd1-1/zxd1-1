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
dirlst=os.listdir(r'D:\abide\Outputs\cpac\nofilt_noglobal\reho')
df=pd.read_csv(r'D:\abide-master\abide-master\Phenotypic_V1_0b_preprocessed1.csv')
df1=df[['FILE_ID','AGE_AT_SCAN','SEX','func_mean_fd','DX_GROUP']]
lst=[]
for i in range(len(df1['FILE_ID'])):
    if df1['FILE_ID'][i]+'_reho.nii.gz'  in dirlst:#将其中数据齐全的884个取出来
        lst.append(i)
data=df1.iloc[lst]
data.index=range(0,884)
print(data['DX_GROUP'])
feature_matrix=[]
num=0
for item in dirlst:
    num+=1
    img = nib.load(r'D:\abide\Outputs\cpac\nofilt_noglobal\reho\{0}'.format(item))  # 下载niifile文件（其实是提取文件）
    img_fdata = img.get_fdata()
    img_fdata=img_fdata.tolist()
    for i in range(len(img_fdata)):
          for j in  range(len(img_fdata[0])):
              for k in range(len(img_fdata[0][0])):
                  img_fdata[i][j][k]=1-img_fdata[i][j][k]
    CC = gd.CubicalComplex(top_dimensional_cells=img_fdata)
    diag = CC.persistence(homology_coeff_field=2)
    diag_β0intervals = []
    diag_β1intervals = []
    diag_β2intervals = []
    for  item in diag:
        if item[0]==0:
           diag_β0intervals.append(item)
        elif item[0]==1:
           diag_β1intervals.append(item)
        elif item[0]==2:
           diag_β2intervals.append(item)
    num00,num01,num02,num03,num04,num05,num06,num07,num08,num09,num010,num011,num012,num013,num014,num015,num016,\
        num017,num018,num019,num020,num021,num022,num023,num024,num025,num026,num027,num028,num029,num030,num031,num032,\
            num033,num034,num035=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    for item in diag_β0intervals:
        if item[1][0]<=0.6<=item[1][1]:
          num00+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.61<=item[1][1]:
          num01+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.62<=item[1][1]:
           num02+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.63<=item[1][1]:
           num03+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.64<=item[1][1]:
           num04+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.65<=item[1][1]:
           num05+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.66<=item[1][1]:
           num06+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.67<=item[1][1]:
           num07+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.68<=item[1][1]:
           num08+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.69<=item[1][1]:
           num09+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.70<=item[1][1]:
           num010+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.71<=item[1][1]:
           num011+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.72<=item[1][1]:
            num012+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.73<=item[1][1]:
            num013+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.74<=item[1][1]:
            num014+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.75<=item[1][1]:
            num015+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.76<=item[1][1]:
          num016+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.77<=item[1][1]:
          num017+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.78<=item[1][1]:
           num018+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.79<=item[1][1]:
           num019+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.80<=item[1][1]:
           num020+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.81<=item[1][1]:
           num021+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.82<=item[1][1]:
           num022+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.83<=item[1][1]:
           num023+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.84<=item[1][1]:
           num024+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.85<=item[1][1]:
           num025+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.86<=item[1][1]:
           num026+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.87<=item[1][1]:
           num027+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.88<=item[1][1]:
            num028+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.89<=item[1][1]:
            num029+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.90<=item[1][1]:
            num030+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.91<=item[1][1]:
            num031+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.92<=item[1][1]:
            num032+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.93<=item[1][1]:
            num033+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.94<=item[1][1]:
            num034+=1
    for item in diag_β0intervals:
        if item[1][0]<=0.95<=item[1][1]:
            num035+=1
    num10, num11, num12, num13, num14, num15, num16, num17, num18, num19, num110, num111, num112, num113, num114, num115 ,\
        num116,num117,num118,num119,num120,num121,num122,num123,num124,num125,num126,num127,num128,num129,num130,num131,num132\
            = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    for item in diag_β1intervals:
        if item[1][0]<=0.65<=item[1][1]:
                num10+=1
    for item in diag_β1intervals:
        if item[1][0] <= 0.66 <= item[1][1]:
                num11 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.67 <= item[1][1]:
                num12 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.68 <= item[1][1]:
                num13 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.69 <= item[1][1]:
                num14 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.70<= item[1][1]:
                num15 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.71 <= item[1][1]:
                num16 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.72 <= item[1][1]:
                num17 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.73 <= item[1][1]:
                num18 += 1
    for item in diag_β1intervals:
         if item[1][0] <= 0.74 <= item[1][1]:
                num19 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.75<= item[1][1]:
                num110 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.76 <= item[1][1]:
                num111 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.77 <= item[1][1]:
                num112 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.78 <= item[1][1]:
                num113 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.79<= item[1][1]:
                num114 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.80 <= item[1][1]:
                num115 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.81<= item[1][1]:
                num116 += 1
    for item in diag_β1intervals:
            if item[1][0] <= 0.82 <= item[1][1]:
                num117 += 1
    for item in diag_β1intervals:
            if item[1][0] <= 0.83 <= item[1][1]:
                num118 += 1
    for item in diag_β1intervals:
            if item[1][0] <= 0.84<= item[1][1]:
                num119 += 1
    for item in diag_β1intervals:
            if item[1][0] <= 0.85 <= item[1][1]:
                num120 += 1
    for item in diag_β1intervals:
            if item[1][0] <= 0.86<= item[1][1]:
                num121 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.87 <= item[1][1]:
                num122 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.88<= item[1][1]:
                num123 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.89<= item[1][1]:
                 num124 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.90 <= item[1][1]:
                num125+= 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.91 <= item[1][1]:
                num126 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.92 <= item[1][1]:
                num127 += 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.93<= item[1][1]:
                num128+= 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.94<= item[1][1]:
                num129 += 1
    for item in diag_β1intervals:
            if item[1][0] <= 0.95 <= item[1][1]:
                num130+= 1
    for item in diag_β1intervals:
        if item[1][0] <= 0.96<= item[1][1]:
                num131 += 1
    for item in diag_β1intervals:
            if item[1][0] <= 0.97 <= item[1][1]:
                num132+= 1
    num20, num21, num22, num23, num24, num25, num26, num27, num28, num29, num210, num211, num212, num213, num214, num215, \
        num216, num217= 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for item in diag_β2intervals:
            if item[1][0] <= 0.8<=item[1][1]:
                num20 += 1
    for item in diag_β2intervals:
            if item[1][0] <= 0.81 <= item[1][1]:
                num21 += 1
    for item in diag_β2intervals:
            if item[1][0] <= 0.82 <= item[1][1]:
                num22 += 1
    for item in diag_β2intervals:
            if item[1][0] <= 0.83 <= item[1][1]:
                num23 += 1
    for item in diag_β2intervals:
            if item[1][0] <= 0.84 <= item[1][1]:
                num24 += 1
    for item in diag_β2intervals:
            if item[1][0] <= 0.85 <= item[1][1]:
                num25 += 1
    for item in diag_β2intervals:
            if item[1][0] <= 0.86 <= item[1][1]:
                num26 += 1
    for item in diag_β2intervals:
            if item[1][0] <= 0.87 <= item[1][1]:
                num27 += 1
    for item in diag_β2intervals:
            if item[1][0] <= 0.88 <= item[1][1]:
                num28 += 1
    for item in diag_β2intervals:
            if item[1][0] <= 0.89 <= item[1][1]:
                num29 += 1
    for item in diag_β2intervals:
            if item[1][0] <= 0.90 <= item[1][1]:
                num210 += 1
    for item in diag_β2intervals:
            if item[1][0] <= 0.91 <= item[1][1]:
                num211 += 1
    for item in diag_β2intervals:
            if item[1][0] <= 0.92 <= item[1][1]:
                num212 += 1
    for item in diag_β2intervals:
            if item[1][0] <= 0.93 <= item[1][1]:
                num213 += 1
    for item in diag_β2intervals:
            if item[1][0] <= 0.94 <= item[1][1]:
                num214 += 1
    for item in diag_β2intervals:
            if item[1][0] <= 0.95 <= item[1][1]:
                num215 += 1
    for item in diag_β2intervals:
            if item[1][0] <= 0.96 <= item[1][1]:
                num216 += 1
    for item in diag_β2intervals:
            if item[1][0] <= 0.97 <= item[1][1]:
                num217 += 1
    print('---------------------{0}-------------------------------'.format(num))
    feature_matrix.append([num00,num01,num02,num03,num04,num05,num06,num07,num08,num09,num010,num011,num012,num013,num014,num015,num016,
        num017,num018,num019,num020,num021,num022,num023,num024,num025,num026,num027,num028,num029,num030,num031,num032,
            num033,num034,num035,num10, num11, num12, num13, num14, num15, num16, num17, num18, num19, num110, num111, num112, num113, num114, num115,
        num116,num117,num118,num119,num120,num121,num122,num123,num124,num125,num126,num127,num128,num129,num130,num131,num132,num20, num21, num22, num23, num24, num25, num26, num27, num28, num29, num210, num211, num212, num213, num214, num215,
        num216, num217])
print(feature_matrix)
np.savetxt('feature_matrix_87',np.array(feature_matrix),encoding='utf-8')
feature_matrix=np.loadtxt('feature_matrix_87')
feature_matrix.tolist()
feature_matrix=pd.DataFrame(feature_matrix,columns=["feature_"+str(i+1) for i in range(87)],index=["subject_"+str(i+1) for i in range(884)])
feature_matrix.to_excel('feature_matrix_87.xlsx')