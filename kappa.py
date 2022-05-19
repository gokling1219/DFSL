import numpy as np

def kappa(testData, k): #testData表示要计算的数据，k表示数据矩阵的是k*k的
    dataMat = np.mat(testData)
    s = dataMat.sum()
    #print(dataMat.shape)
    print(dataMat)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i]*1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    #xsum是个k行1列的向量，ysum是个1行k列的向量
    #Pe = float(ysum * xsum) / float(s * 1.0) / float(s * 1.0)
    Pe = float(ysum * xsum) / float(s ** 2)
    print("Pe = ", Pe)
    P0 = float(P0/float(s*1.0))
    #print("P0 = ", P0)
    cohens_coefficient = float((P0-Pe)/(1-Pe))

    a = []
    a = dataMat.sum(axis=0)
    a = np.float32(a)
    a = np.array(a)
    a = np.squeeze(a)

    print(a)

    for i in range(k):
        #print(dataMat[i, i])
        a[i] = float(dataMat[i, i]*1.0)/float(a[i]*1.0)
    print(a*100)
    #print(a.shape)
    print("AA: ", a.mean()*100)
    return cohens_coefficient, a.mean()*100, a*100

'''
# 3D_L2
testData = [6558, 0, 14, 0, 0, 0, 33, 24, 2,
            0, 18551, 0, 6, 0, 92, 0, 0 , 0,
            0, 0, 2093, 1, 0, 0, 0, 5, 0,
            21, 3, 7, 3005, 0, 4, 4, 20, 0,
            0, 0, 0, 0, 1345, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 5028, 0, 0, 0,
            2, 0, 0, 0, 0, 0, 1328, 0, 0,
            13, 1, 15, 13, 0, 0, 0, 3640, 0,
            0, 0, 0, 3, 0, 0, 0, 0, 944]
testData = np.array(testData).reshape(9,9)
print(kappa(testData,9))
'''

'''
# 3D
testData = [6421, 0, 29, 2, 0, 0, 43, 133, 3,
            2, 18486, 0, 92, 0, 46, 0, 23, 0,
            9, 0, 2060, 1, 0, 0, 1, 27, 1,
            6, 0, 0, 3054, 0, 0, 0, 3, 1,
            0, 0, 0, 0, 1345, 0, 0, 0, 0,
            0, 9, 0, 0, 0, 5020, 0, 0, 0,
            21, 0, 0, 0, 0, 0, 1309, 0, 0,
            1, 7, 20, 30, 0, 1, 0, 3623, 0,
            2, 0, 0, 0, 0, 0, 0, 0, 945]
testData = np.array(testData).reshape(9,9)
print("kappa = ", kappa(testData,9))
'''



