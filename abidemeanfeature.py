import numpy as np
import pandas as pd

data=np.loadtxt('feature_matrix_87')
data=data.tolist()
lst=[]
for i in range(87):
    a=0
    for j in range(884):
        a+=data[j][i]
    lst.append(a/884)
print(lst)
for i in range(87):
    for j in range(884):
        data[j][i]=data[j][i]-lst[i]
#print(data)
data=pd.DataFrame(data,columns=["feature_"+str(i+1) for i in range(87)],index=["subject_"+str(i+1) for i in range(884)])

data.to_excel('abidemeanfeaturedata.xlsx')