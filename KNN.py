import numpy as np
import operator

def dataset():
    group=np.array(([1,1.1],[1,1],[0,0],[0,0.1]))
    lables=['A','A','B','B']
    return group,lables
def knn(inx,dataSet,lables,k):
    dataSetSize=dataSet.shape[0]
    _dataSet=np.mat(dataSet)
    _dataSet-=inx
    dataDistance=np.zeros(dataSetSize)
    for i in range(dataSetSize):
        for j in range(_dataSet.shape[1]):
            dataDistance[i]+=_dataSet[i,j]**2
        dataDistance[i]=dataDistance[i]**0.5
    indexSorted=dataDistance.argsort()
    predict={}
    for i in range(k):
        votelable=lables[indexSorted[i]]
        predict[votelable]=predict.get(votelable,0)+1
    res=sorted(predict.items(),key=lambda x: x[0],reverse=True)
    return res[0][0]

group,lables=dataset()
res=knn([0,0],group,lables,3)
print(res)