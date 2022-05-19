import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def data_read(filepath):
    fp = open(filepath, "r")
    datas = []  # 存储处理后的数据
    lines = fp.readlines()  # 读取整个文件数据
    i = 0  # 为一行数据
    for line in lines:
        row = line.strip('\n').split()  # 去除两头的换行符，按空格分割
        datas.append(row)
        i = i + 1
        # print("读取第", i, "行")

    fp.close()
    return datas

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
    Pe = float(ysum * xsum) / float(s * 1.0) / float(s * 1.0)
    #print("Pe = ", Pe)
    P0 = float(P0/float(s*1.0))
    #print("P0 = ", P0)
    cohens_coefficient = float((P0-Pe)/(1-Pe))

    a = []
    a = dataMat.sum(axis=0)
    a = np.float32(a)
    a = np.array(a)
    a = np.squeeze(a)

    #print(a)

    for i in range(k):
        #print(dataMat[i, i])
        a[i] = float(dataMat[i, i]*1.0)/float(a[i]*1.0)
    print('各类别精度： ', a)
    print("AA: ", a.mean())
    return cohens_coefficient

def class_map_PU(final_result, num_labeled):

    #colors = ['black', 'gray', 'lime', 'cyan', 'forestgreen', 'hotpink', 'saddlebrown', 'purple', 'red', 'yellow']
    colors = ['black', 'saddlebrown', 'purple', 'red', 'yellow']
    cmap = mpl.colors.ListedColormap(colors)
    plt.xticks([])
    plt.yticks([])

    gci = plt.imshow(final_result, cmap=cmap)
    cbar = plt.colorbar(gci)
    cbar.set_ticks(np.linspace(0.5, 8.5, 10))
    cbar.set_ticklabels(('Unlabeled', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                         'Metal Sheets', 'Bare Soil', 'Bitumen', 'Bricks', 'Shadow'
                         ))

    plt.imshow(final_result, cmap=cmap)
    plt.savefig('./class_map_PU_DFSL_' + str(num_labeled) + '_5-8.png', format='png', dpi=1000)
    #plt.savefig('./class_map_PU_DFSL_' + str(num_labeled) + '_5-8.eps', format='eps')
    plt.show()

def class_map_IP(final_result, num_labeled):

    colors = ['black', 'gray', 'lime', 'cyan', 'forestgreen', 'hotpink', 'saddlebrown', 'purple', 'red', 'yellow']
    cmap = mpl.colors.ListedColormap(colors)
    plt.xticks([])
    plt.yticks([])

    gci = plt.imshow(final_result, cmap=cmap)
    cbar = plt.colorbar(gci)
    cbar.set_ticks(np.linspace(0.5, 8.5, 10))
    cbar.set_ticklabels(('Unlabeled', 'Corn-notill', 'Corn-mintill', 'Grass-pasture', 'Grass-trees',
                         'Hay-windrowed', 'Soybean-notill', 'Soybean-mintill', 'Soybean-clean', 'Woods'
                         ))

    plt.imshow(final_result, cmap=cmap)
    plt.savefig('./results/class_map_IP_DFSL_' + str(num_labeled) + '.png', format='png')
    plt.savefig('./results/class_map_IP_DFSL_' + str(num_labeled) + '.eps', format='eps')
    plt.show()

def class_map_SA(final_result, num_labeled):

    #colors = ['black', 'gray', 'lime', 'cyan', 'forestgreen', 'hotpink', 'saddlebrown', 'purple', 'red', 'yellow']
    colors = ['black',  'red', 'yellow', 'blue', 'aqua', 'olive', 'sandybrown', 'lawngreen']
    cmap = mpl.colors.ListedColormap(colors)
    plt.xticks([])
    plt.yticks([])

    gci = plt.imshow(final_result, cmap=cmap)
    cbar = plt.colorbar(gci)
    cbar.set_ticks(np.linspace(0.5, 8.5, 10))
    cbar.set_ticklabels(('Unlabeled', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                         'Metal Sheets', 'Bare Soil', 'Bitumen', 'Bricks', 'Shadow'
                         ))

    plt.imshow(final_result, cmap=cmap)
    plt.savefig('./SA_DFSL_' + str(num_labeled) + '_0-6.png', format='png', dpi=1000)
    #plt.savefig('./class_map_PU_DFSL_' + str(num_labeled) + '_5-8.eps', format='eps')
    plt.show()

def class_map_KSC(final_result, num_labeled):

    #colors = ['black', 'gray', 'lime', 'cyan', 'forestgreen', 'hotpink', 'saddlebrown', 'purple', 'red', 'yellow']
    colors = ['black', 'red', 'yellow', 'blue', 'aqua', 'olive', 'sandybrown']
    cmap = mpl.colors.ListedColormap(colors)
    plt.xticks([])
    plt.yticks([])

    gci = plt.imshow(final_result, cmap=cmap)
    cbar = plt.colorbar(gci)
    cbar.set_ticks(np.linspace(0.5, 8.5, 10))
    cbar.set_ticklabels(('Unlabeled', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                         'Metal Sheets', 'Bare Soil', 'Bitumen', 'Bricks', 'Shadow'
                         ))

    plt.imshow(final_result, cmap=cmap)
    plt.savefig('./KSC_DFSL_' + str(num_labeled) + '_7-12.png', format='png', dpi=1000)
    #plt.savefig('./class_map_PU_DFSL_' + str(num_labeled) + '_5-8.eps', format='eps')
    plt.show()