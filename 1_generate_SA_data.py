import numpy as np
import h5py
from sklearn.decomposition import PCA
from scipy.io import loadmat

def PCA_transform_img(img=None, n_principle=3):

    height = img.shape[0]
    width = img.shape[1]
    dim = img.shape[2]
    # reshape img, HORIZONTALLY strench the img, without changing the spectral dim.
    reshaped_img = img.reshape(height * width, dim)

    pca = PCA(n_components=n_principle) # 保留下来的特征个数n
    pca_img = pca.fit_transform(reshaped_img) # 用reshaped_img来训练PCA模型，同时返回降维后的数据
                  # shape (n_samples, n_features)

    # Regularization: Think about energy of each principles here.
    reg_img = pca_img * 1.0 / pca_img.max()
    # print(reg_img.shape)  (207400, 3)
    return reg_img

def Patch(data,height_index,width_index,PATCH_SIZE):   # PATCH_SIZE为一个patch（边长-1）的一半    data维度(H,W,C)
    height_slice = slice(height_index-PATCH_SIZE, height_index+PATCH_SIZE+1)
    width_slice = slice(width_index-PATCH_SIZE, width_index+PATCH_SIZE+1)
    # 由height_index和width_index定位patch中心像素
    patch = data[height_slice, width_slice,:]
    patch = patch.reshape(-1,patch.shape[0]*patch.shape[1]*patch.shape[2])
    # print(patch.shape)                  #为一行  (1, 243) 243 = 9*9*3
    return patch

img = loadmat('d:\hyperspectral_data\Salinas_corrected.mat')['salinas_corrected']
gt = loadmat('d:\hyperspectral_data\Salinas_gt.mat')['salinas_gt']
# print(img.shape)  #(610, 340, 103)

#img = PCA_transform_img(img, 3)
#img = img.reshape(610, 340, 3)   # 重新reshape成三维

img = img[:, :, 0:100] # 只选取前100个波段

img = ( img * 1.0 - img.min() ) / ( img.max() - img.min() )

[m, n, b] = img.shape
label_num = gt.max()  #最大为9，即除0外包括9类
PATCH_SIZE = 4   #每一个patch边长大小为9

# padding the hyperspectral images
img_temp = np.zeros((m + 2 * PATCH_SIZE, n + 2 * PATCH_SIZE, b), dtype=np.float32)
img_temp[PATCH_SIZE:(m + PATCH_SIZE), PATCH_SIZE:(n + PATCH_SIZE), :] = img[:, :, :]

for i in range(PATCH_SIZE):
    img_temp[i, :, :] = img_temp[2 * PATCH_SIZE - i, :, :]
    img_temp[m + PATCH_SIZE + i, :, :] = img_temp[m + PATCH_SIZE - i - 2, :, :]

for i in range(PATCH_SIZE):
    img_temp[:, i, :] = img_temp[:, 2 * PATCH_SIZE - i, :]
    img_temp[:, n + PATCH_SIZE + i, :] = img_temp[:, n + PATCH_SIZE  - i - 2, :]

img = img_temp

gt_temp = np.zeros((m + 2 * PATCH_SIZE, n + 2 * PATCH_SIZE), dtype=np.int8)
gt_temp[PATCH_SIZE:(m + PATCH_SIZE), PATCH_SIZE:(n + PATCH_SIZE)] = gt[:, :]
gt = gt_temp

[m, n, b] = img.shape
# count = 0 #统计有多少个中心像素类别不为0的patch


def preparation():
    f = open('./SA_gt_index.txt', 'w')
    f1 = open('./SA_label.txt', 'w')
    data = []
    label = []

    for i in range(PATCH_SIZE, m - PATCH_SIZE):
        for j in range(PATCH_SIZE, n - PATCH_SIZE):
            if gt[i, j] == 0:
                continue
            else:
                # count += 1
                temp_data = Patch(img, i, j, PATCH_SIZE)
                # temp_label = np.zeros((1, label_num), dtype=np.int8)  # temp_label为一行九列[0,1,2,....,7,8]表示类别
                # temp_label[0, gt[i, j] - 1] = 1
                temp_label = gt[i, j] - 1  # 直接用0-8表示，不用独热编码
                data.append(temp_data)  # 每一行表示一个patch
                label.append(temp_label)
                gt_index = ((i - PATCH_SIZE) * 217 + j - PATCH_SIZE)  # 记录坐标，用于可视化分类预测结果
                f.write(str(gt_index) + '\n')
                f1.write(str(temp_label) + '\n')

    # print(count)  #42776

    data = np.array(data)
    print(data.shape)  # (42776, 1, 867)
    data = np.squeeze(data)
    print("squeeze : ", data.shape)  # squeeze :  (42776, 867)
    label = np.array(label)
    print(label.shape)  # (42776, 1, 9)
    label = np.squeeze(label)
    print("squeeze : ", label.shape)  # squeeze :  (42776, 9)
    print(np.unique(label))

    f = h5py.File('.\SA9_9_100_labeled.h5', 'w')
    f['data'] = data
    f['label'] = label
    f.close()

preparation()