import numpy as np
import scipy.io as scio
from scipy import stats
import pandas as pd
import statistics
a=[0.8710407239819005, 1.0746606334841629, 1.3009049773755657, 1.5520361990950227, 1.9015837104072397, 2.2782805429864252, 2.7205882352941178,
 3.254524886877828, 3.816742081447964, 4.5542986425339365, 5.300904977375565, 6.021493212669683, 6.644796380090498, 7.619909502262443,
 8.406108597285067, 9.408371040723981, 10.333710407239819, 11.19343891402715, 11.845022624434389, 12.80316742081448, 13.369909502262443,
 13.833710407239819, 14.106334841628959, 14.441176470588236, 14.703619909502262, 14.940045248868778, 15.31787330316742, 15.661764705882353,
 15.700226244343892, 15.144796380090497, 13.59049773755656, 11.342760180995475, 8.126696832579185, 5.04524886877828, 2.425339366515837,
 1.1266968325791855, 0.06334841628959276, 0.0746606334841629, 0.1074660633484163, 0.14705882352941177, 0.20475113122171945, 0.2409502262443439,
 0.3438914027149321, 0.5113122171945701, 0.6900452488687783, 0.9423076923076923, 1.2850678733031675, 1.7002262443438914, 2.2047511312217196,
 2.680995475113122, 3.4592760180995477, 4.33710407239819, 5.399321266968326, 6.794117647058823, 8.345022624434389, 9.841628959276019,
 11.54185520361991, 13.036199095022624, 14.54524886877828, 15.864253393665159, 16.835972850678733, 17.27262443438914, 17.828054298642535,
 19.25339366515837, 19.463800904977376, 16.986425339366516, 7.452488687782806, 1.6357466063348416, 2.001131221719457, 0.0418552036199095,
 0.05656108597285068, 0.08823529411764706, 0.1244343891402715, 0.20588235294117646, 0.3608597285067873, 0.5780542986425339, 0.8812217194570136,
1.2794117647058822, 1.8880090497737556, 2.7070135746606336, 3.524886877828054, 4.150452488687783, 4.200226244343892, 5.3789592760181,
9.365384615384615, 4.085972850678733, 1.7024886877828054]
# c=[]
# for i in range(87):
#    c.append(a[i]+b[i])
# print(c)
# data=np.loadtxt('meanfeaturedata')
# data=data.tolist()
# for i in range(87):
#     for j in range(884):
#         data[j][i]=data[j][i]+c[i]
# #print(data)
# np.savetxt('backfeaturedata',data)
path=r'matlab1mat'
matdata=scio.loadmat(path)
print(matdata['Ddata'])
data=matdata['Ddata'].tolist()
for i in range(87):
    for j in range(884):
        data[j][i]=data[j][i]+a[i]
print(np.asarray(data))
print('------------------------------')
np.savetxt('add_back_the_mean_after_combat',data)
data1=np.loadtxt('add_back_the_mean_after_combat')
print(data1)
c=[]
data1=data1.tolist()
for i in range(87):
   m=0
   for j in range(884):
       m+= data1[j][i]
   c.append(m/884)
print(c)
from numpy.linalg import inv  # 矩阵求逆
from numpy import dot  # 求矩阵点乘



txt = np.loadtxt('add_back_the_mean_after_combat')
dataset = pd.DataFrame(txt)
print(dataset)
# txtDF.to_csv('file.csv', index=False)
# data = pd.read_csv('file.csv')
#dataset = pd.DataFrame(data,copy = True)
data=pd.read_excel('data.xlsx')
print(type(data))
Y1= data['AGE_AT_SCAN']
Y=[]
for i in range(884):
    Y.append(Y1[i])
mean_Y=np.mean(Y)
#for column in dataset.columns:
lst1,lst2=[],[]
for i in range(87):
    X=[]
    for j in range(884):
         X.append(dataset[i][j])

    up=0
    down1=0
    for k in range(884):
        up+=X[k]*(Y[k]-mean_Y)
        down1+=(Y[k]**2)

    w=up/(down1-(((np.sum(Y))**2)/884))
    b=0
    for l in range(884):
        b+= (X[l]-w*Y[l])

    lst1.append(w)
    lst2.append(b/884)
print(lst1)
print(lst2)
residual_numpy=np.zeros((884,87))
residual=residual_numpy.tolist()
for i in range(87):
    for j in range(884):
        residual[j][i]=dataset[i][j]-lst1[i]*Y1[j]-lst2[i]
print(np.asarray(residual))
np.savetxt('residual_matrix_with_age',np.asarray(residual))

# a=np.loadtxt('residual_matrix_with_age')
# a=a.tolist()
# alst=[]
# for i in range(85):
#     num=0
#     for j in range(884):
#         if a[j][i]<0:
#             num+=1
#     alst.append(num)
# print(alst)
# num2=0
# for i in alst:
#     if i <=442:
#         num2+=1
# print(num2)
for i in range(87):
    asdlst = []
    tdlst=[]
    num=0
    for j in range(884):
        if data['DX_GROUP'][j]==1:
            asdlst.append(residual[j][i])
        else:
            tdlst.append(residual[j][i])
    mean1,mean2=0,0
    for k in asdlst:
            mean1+=k
    for l in tdlst:
            mean2+=l
    print('ASD特征均值:',mean1/408)
    print('ASD特征标准差:',statistics.stdev(asdlst))
    print('TD特征均值:',mean2/476)
    print('TD特征标准差:',statistics.stdev(tdlst))
    print(i,stats.levene(asdlst,tdlst))
    statistic, pvalue=stats.levene(asdlst,tdlst)
    if pvalue>0.05:
        print(i,stats.ttest_ind(asdlst, tdlst, equal_var=True))
    else:
        print(i, stats.ttest_ind(asdlst, tdlst, equal_var=False))
    print('--------------------------------------------')






