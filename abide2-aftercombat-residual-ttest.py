import numpy as np
import scipy.io as scio
from scipy import stats
import pandas as pd
import statistics

a=[79.39245283018867, 94.09685534591195, 110.19119496855346, 127.24528301886792, 145.53081761006288, 163.9433962264151, 183.07672955974843,
   201.58867924528303, 217.337106918239, 228.98238993710692, 234.4566037735849, 232.63396226415094, 221.737106918239, 201.95849056603774,
   176.61509433962263, 147.7194968553459, 117.4314465408805, 89.37232704402516, 64.72327044025157, 44.633962264150945, 30.028930817610064,
   19.333333333333332, 12.161006289308176, 7.589937106918239, 4.633962264150943, 3.0641509433962266, 22.872955974842768, 32.534591194968556,
   45.46666666666667, 62.880503144654085, 86.09308176100629, 117.26540880503144, 157.34088050314466, 208.96981132075473, 272.27672955974845,
   347.8490566037736, 432.36603773584903, 521.6012578616352, 609.6163522012579, 684.9698113207547, 736.7949685534592, 748.0679245283019,
   713.7509433962264, 634.2025157232705, 515.4918238993711, 387.8691823899371, 269.22767295597487, 175.28679245283018, 107.70314465408805,
   65.48553459119496, 38.61132075471698, 22.29056603773585, 17.316981132075473, 25.983647798742137, 38.378616352201256, 55.56981132075472,
   79.62138364779874, 111.1685534591195, 151.3685534591195, 202.62138364779875, 261.9132075471698, 320.5899371069182, 367.13459119496855,
   390.2503144654088, 380.5056603773585, 339.3798742138365, 281.4327044025157, 218.07798742138365, 159.5572327044025, 110.65031446540881,
   73.67295597484276, 47.09308176100629, 28.452830188679247]

path=r'matlab2.mat'
matdata=scio.loadmat(path)
print(matdata['Ddata'])
data=matdata['Ddata'].tolist()
for i in range(73):
    for j in range(795):
        data[j][i]=data[j][i]+a[i]
print(np.asarray(data))
print('------------------------------')
np.savetxt('abide2combat后将均值加回来',data)
data1=np.loadtxt('abide2combat后将均值加回来')
print(data1)
c=[]
data1=data1.tolist()
for i in range(73):
   m=0
   for j in range(795):
       m+= data1[j][i]
   c.append(m/795)
print(c)




txt = np.loadtxt('abide2combat后将均值加回来')
dataset = pd.DataFrame(txt)
print(dataset)

data=pd.read_excel('abide2_information.xlsx')
print(type(data))
Y1= data['AGE_AT_SCAN ']
Y=[]
for i in range(795):
    Y.append(Y1[i])
mean_Y=np.mean(Y)
#for column in dataset.columns:
lst1,lst2=[],[]
for i in range(73):
    X=[]
    for j in range(795):
         X.append(dataset[i][j])

    up=0
    down1=0
    for k in range(795):
        up+=X[k]*(Y[k]-mean_Y)
        down1+=(Y[k]**2)

    w=up/(down1-(((np.sum(Y))**2)/795))
    b=0
    for l in range(795):
        b+= (X[l]-w*Y[l])

    lst1.append(w)
    lst2.append(b/795)
print(lst1)
print(lst2)
residual_numpy=np.zeros((795,73))
residual=residual_numpy.tolist()
for i in range(73):
    for j in range(795):
        residual[j][i]=dataset[i][j]-lst1[i]*Y1[j]-lst2[i]
print(np.asarray(residual))
np.savetxt('abide2residual_matrix和年龄回归',np.asarray(residual))

for i in range(73):
    asdlst = []
    tdlst=[]
    num=0
    for j in range(795):
        if data['DX_GROUP'][j]==1:
            asdlst.append(residual[j][i])
        else:
            tdlst.append(residual[j][i])
    mean1,mean2=0,0
    for k in asdlst:
            mean1+=k
    for l in tdlst:
            mean2+=l
    print('ASD特征均值:',mean1/342)
    print('ASD特征标准差:',statistics.stdev(asdlst))
    print('TD特征均值:',mean2/453)
    print('TD特征标准差:',statistics.stdev(tdlst))
    print(i,stats.levene(asdlst,tdlst))
    statistic, pvalue=stats.levene(asdlst,tdlst)
    if pvalue>0.05:
        print(i,stats.ttest_ind(asdlst, tdlst, equal_var=True))
    else:
        print(i, stats.ttest_ind(asdlst, tdlst, equal_var=False))
    print('--------------------------------------------')