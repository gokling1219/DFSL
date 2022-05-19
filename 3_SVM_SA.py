import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.model_selection import GridSearchCV
import joblib
from kappa import kappa
from scipy.io import loadmat
from sklearn import metrics
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import input_data_SA

num_labeled = 5

# 加载数据


# 独热编码转换为类别标签
#train_label=np.argmax(train_label,1)
#print(train_label)
#test_label=np.argmax(test_label,1)
#print(test_label)
#print(np.unique(test_label))


################################################### KNN训练及分类 ###################################################
OA_list = [0,0,0,0,0,0,0,0,0,0]
AA_list = [0,0,0,0,0,0,0,0,0,0]
kappa_list = [0,0,0,0,0,0,0,0,0,0]
CA_list = np.zeros((10,16))

n = 54129

seed_number = [1,2,3,4,5,6,7,8,9,10]
#seed_number = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]

for INDEX in range(10):

    np.random.seed(seed_number[INDEX])

    mnist = input_data_SA.read_data_sets(n_labeled=num_labeled)
    train_data = mnist.train.images
    train_label = mnist.train.labels
    test_data = mnist.test.images
    test_label = mnist.test.labels

    start = time.time()
    C = np.logspace(-2, 8, 11, base=2)  # 2为底数，2的-2次方到2的8次方，一共11个数
    gamma = np.logspace(-2, 8, 11, base=2)

    parameters = {'C': C,
                  'gamma': gamma}
    # 问题：参数设置规律？？？

    clf = GridSearchCV(svm.SVC(kernel='rbf'), parameters, cv=4, refit=True)
    # Refit an estimator using the best found parameters on the whole dataset.

    clf.fit(train_data, train_label)

    print("time = ", time.time() - start)

    # 输出最佳参数组合
    # print('随机搜索-度量记录：',clf.cv_results_)  # 包含每次训练的相关信息
    print('随机搜索-最佳度量值:', clf.best_score_)  # 获取最佳度量值
    print('随机搜索-最佳参数：', clf.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
    print('随机搜索-最佳模型：', clf.best_estimator_)  # 获取最佳度量时的分类器模型

    # 存储结果学习模型，方便之后的调用
    joblib.dump(clf.best_estimator_, r'.\SA_BEST_MODEL_' + str(INDEX + 1) + '.m')

    clf = joblib.load(r".\SA_BEST_MODEL_" + str(INDEX + 1) + ".m")

    start = time.time()
    predict_label = clf.predict(test_data)  # (42776,)

    matrix = metrics.confusion_matrix(test_label, predict_label)
    print(matrix)
    print('OA = ', np.sum(np.trace(matrix)) / float(n) * 100)
    OA_list[INDEX] = np.sum(np.trace(matrix)) / float(n) * 100

    kappa_temp, aa_temp, ca_temp = kappa(matrix, 16)
    AA_list[INDEX] = aa_temp
    CA_list[INDEX] = ca_temp
    kappa_list[INDEX] = kappa_temp * 100
    print("kappa = ", kappa_temp * 100)

    gt = loadmat('d:\hyperspectral_data\Salinas_gt.mat')['salinas_gt']

    # 将预测的结果匹配到图像中
    new_show = np.zeros((gt.shape[0], gt.shape[1]))
    k = 0
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i][j] != 0:
                new_show[i][j] = predict_label[k]
                new_show[i][j] += 1
                k += 1

    np.save("SA/SA_" + str(np.sum(np.trace(matrix)) / float(n) * 100) + "npy", new_show)
    # print new_show.shape

    # 展示地物

    colors = ['black', 'gray', 'lime', 'cyan', 'forestgreen', 'hotpink', 'saddlebrown',
              'purple', 'red', 'yellow', 'blue', 'steelblue', 'olive', 'sandybrown', 'lawngreen', 'darkorange',
              'whitesmoke']

    cmap = mpl.colors.ListedColormap(colors)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(new_show, cmap=cmap)
    plt.savefig("SA/SA_dfslsvm_" + str(np.sum(np.trace(matrix)) / float(n)) + "_" + str(INDEX + 1) + ".png", dpi=1000)  # 保存图像
    #plt.show()
    print("time = ", time.time() - start)

print("\n")
print("OA_list", OA_list)
print(np.array(OA_list).mean())
print(np.array(OA_list).std())
print("AA_list", AA_list)
print(np.array(AA_list).mean())
print(np.array(AA_list).std())
print("kappa_list", kappa_list)
print(np.array(kappa_list).mean())
print(np.array(kappa_list).std())
print("CA_list", CA_list)
print(np.array(CA_list).mean(axis=0))
print(np.array(CA_list).std(axis=0))

f = open('SA/SA_results.txt', 'w')
for index in range(CA_list.mean(axis=0).shape[0]):
    f.write(str(np.array(CA_list).mean(axis=0)[index]) + '\n')
f.write(str(np.array(OA_list).mean()) + '\n')
f.write(str(np.array(AA_list).mean()) + '\n')
f.write(str(np.array(kappa_list).mean()) + '\n')
f.write("\n\n\n")
for index in range(CA_list.std(axis=0).shape[0]):
    f.write(str(np.array(CA_list).std(axis=0)[index]) + '\n')
f.write(str(np.array(OA_list).std()) + '\n')
f.write(str(np.array(AA_list).std()) + '\n')
f.write(str(np.array(kappa_list).std()) + '\n')