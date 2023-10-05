import numpy as np
import pandas as pd

data=np.loadtxt('abide2feature_matrix_73')
data=data.tolist()
lst=[]
for i in range(73):
    a=0
    for j in range(795):
        a+=data[j][i]
    lst.append(a/795)
print(lst)
for i in range(73):
    for j in range(795):
        data[j][i]=data[j][i]-lst[i]
#print(data)
data=pd.DataFrame(data,columns=["feature_"+str(i+1) for i in range(73)],index=["subject_"+str(i+1) for i in range(795)])
np.savetxt('abide2meanfeaturedata',np.asarray(data))
print(lst)
data.to_excel('abide2meanfeaturedata.xlsx')



