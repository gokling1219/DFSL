from __future__ import print_function
import h5py
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os, time,sys

#os.environ["CUDA_VISIBLE_DEVICES"] = "5"

#sys.setrecursionlimit(2000000)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')  # conv2d, [1, 1, 1, 1]

def max_pool_3x3(x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
    #return tf.nn.max_pool3d(x, ksize=[1, 4, 2, 2, 1], strides=[1, 4, 2, 2, 1], padding='SAME')
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 4, 1], strides=[1, 2, 2, 4, 1], padding='SAME')


num = 8
num1 = 2 * num
num2 = 2 * num1

# res1
W_conv1_1 = weight_variable([3, 3, 3, 1, num])
b_conv1_1 = bias_variable([num])

W_conv1_2 = weight_variable([3, 3, 3, num, num])
b_conv1_2 = bias_variable([num])

W_conv1_3 = weight_variable([3, 3, 3, num, num])
b_conv1_3 = bias_variable([num])

W_conv1 = weight_variable([3, 3, 3, num, num])
b_conv1 = bias_variable([num])

# res2
W_conv2_1 = weight_variable([3, 3, 3, num, num1])
b_conv2_1 = bias_variable([num1])

W_conv2_2 = weight_variable([3, 3, 3, num1, num1])
b_conv2_2 = bias_variable([num1])

W_conv2_3 = weight_variable([3, 3, 3, num1, num1])
b_conv2_3 = bias_variable([num1])

W_conv2 = weight_variable([3, 3, 3, num1, num1])
b_conv2 = bias_variable([num1])

# res3
W_conv3 = weight_variable([3, 3, 3, num1, num2])
b_conv3 = bias_variable([num2])



def encoder(x):

    h_conv1_1 = tf.nn.relu(conv3d(x, W_conv1_1) + b_conv1_1)
    h_conv1_2 = tf.nn.relu(conv3d(h_conv1_1, W_conv1_2) + b_conv1_2)
    h_conv1_3 = tf.nn.relu(conv3d(h_conv1_2, W_conv1_3) + b_conv1_3)

    h_conv1 = tf.nn.relu(conv3d(h_conv1_3, W_conv1) + b_conv1)  + h_conv1_1

    # dimension reduction
    h_pool1 = max_pool_3x3(h_conv1)  # 5*5*25

    h_conv2_1 = tf.nn.relu(conv3d(h_pool1, W_conv2_1) + b_conv2_1)
    h_conv2_2 = tf.nn.relu(conv3d(h_conv2_1, W_conv2_2) + b_conv2_2)
    h_conv2_3 = tf.nn.relu(conv3d(h_conv2_2, W_conv2_3) + b_conv2_3)
    h_conv2 = tf.nn.relu(conv3d(h_conv2_3, W_conv2) + b_conv2)  + h_conv2_1

    # dimension reduction
    h_pool2 = max_pool_3x3(h_conv2)  # 3*3*7

    h_conv3 = tf.nn.conv3d(h_pool2, W_conv3, strides=[1, 1, 1, 1, 1], padding='VALID') + b_conv3

    y_conv = tf.reshape(h_conv3, [-1, 5 * num2])

    return y_conv


############################################################ ?????????????????? ############################################################
def euclidean_distance(a, b):
    # a.shape = N x D
    # b.shape = M x D
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    # expand_dims?????????a.shape = N x 1 x D ???tile?????? N x M x D
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    # expand_dims?????????b.shape = 1 x M x D ???tile?????? N x M x D
    return tf.reduce_mean(tf.square(a - b), axis=2) # ??????380*20 ??????????????????D?????????????????????
############################################################ ?????????????????? ############################################################


############################################################ ??????????????? ############################################################
n_epochs = 100
n_episodes = 100
n_way = 20
n_shot = 1
n_query = 19
n_examples = 200 # ???????????????????????????????????????78???
im_width, im_height, channels = 9, 9, 100
############################################################ ??????????????? ############################################################


############################################################ ?????????????????? ############################################################


#x = tf.placeholder(tf.float32, [None, None, channels, im_height, im_width]) # 20, 1, 100, 9, 9
#q = tf.placeholder(tf.float32, [None, None, channels, im_height, im_width]) # 20, 19, 100, 9, 9

x = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels]) # 20, 1, 100, 9, 9
q = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels]) # 20, 19, 100, 9, 9
y = tf.placeholder(tf.int64, [None, None]) # 20, 19
y_one_hot = tf.one_hot(y, depth=n_way) # 20, 19, 20

# ??????few-shot??????
emb_in = encoder( tf.expand_dims( tf.reshape(x, [n_way * n_shot, im_height, im_width, channels]), -1 ) )
#emb_in = encoder( tf.expand_dims( tf.reshape(x, [n_way * n_shot, channels, im_height, im_width]), -1 ) )

emb_dim = tf.shape(emb_in)[-1] # ??????????????????

emb_x = tf.reduce_mean(tf.reshape(emb_in, [n_way, n_shot, emb_dim]), axis=1) # ?????????axis = 1???????????????
# reduce_mean?????????emb_x???????????????????????????????????????????????????????????????????????????????????????????????????????????????n_shot???1???????????????????????????????????????
# ?????????????????????????????????
# ?????? emb_x?????????20, emb_dim

emb_q = encoder( tf.expand_dims( tf.reshape(q, [-1, im_height, im_width, channels]), -1 ) ) # reuse = True ????????????????????????
#emb_q = encoder( tf.expand_dims( tf.reshape(q, [-1, channels, im_height, im_width]), -1 ) ) # reuse = True ????????????????????????
# ?????? emb_q?????????20*19, emb_dim

# ????????????
dists = euclidean_distance(emb_q, emb_x) # ??????????????????????????????????????????????????????
#a = tf.shape(dists) #: [380  20]
log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [n_way, n_query, -1]) # 20???19???20
ce_loss = -tf.reduce_mean( tf.reshape( tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1] ) )
# tf.multiply?????????????????????????????????????????????  ????????????????????????

# ???????????????
acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), y)))


train_op = tf.train.AdamOptimizer(0.001).minimize(ce_loss)
sess = tf.InteractiveSession()
init_op = tf.global_variables_initializer()
sess.run(init_op)

#dim = emb_dim.eval() # ??????CNN??????????????????
# print(dim) 256

############################################################ ?????????????????? ############################################################


def train_CNN():

    f = h5py.File('train_source_data.h5', 'r')
    train_dataset = f['data'][:]
    f.close()
    train_dataset = train_dataset.reshape(-1, n_examples, 9, 9, 100)  # ????????????78????????????200?????????
    #train_dataset = train_dataset.transpose((0, 1, 4, 2, 3))
    n_classes = train_dataset.shape[0]
    print(train_dataset.shape) # (78, 200, 100, 9, 9)

    # ????????????
    a = time.time()
    for ep in range(n_epochs):  # ????????????100???

        for epi in range(n_episodes):

            print(epi)

            epi_classes = np.random.permutation(n_classes)[:n_way]  # ???78???????????????????????????20??? 78??????????????? ????????????20??????????????????15 69 23 ....

            #support = np.zeros([n_way, n_shot, channels, im_height, im_width], dtype=np.float32)  # n_shot = 1
            #query = np.zeros([n_way, n_query, channels, im_height, im_width], dtype=np.float32)  # n_query= 19
            support = np.zeros([n_way, n_shot, im_height, im_width, channels], dtype=np.float32)  # n_shot = 1
            query = np.zeros([n_way, n_query, im_height, im_width, channels], dtype=np.float32)  # n_query= 19
            # ???n_way??????????????????n_shot/n_query??????.....

            for i, epi_cls in enumerate(epi_classes):
                selected = np.random.permutation(n_examples)[:n_shot + n_query]

                support[i] = train_dataset[epi_cls, selected[:n_shot]]
                query[i] = train_dataset[epi_cls, selected[n_shot:]]

            # support = np.expand_dims(support, axis=-1)
            # query = np.expand_dims(query, axis=-1)

            labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)
            # labels????????????????????????????????????????????????
            # labels???20???19????????????0000...0000/1919.....19???????????????0123...19 ???n_query = 19

            _, ls, ac = sess.run([train_op, ce_loss, acc], feed_dict={x: support, q: query, y: labels})

            if epi == 9:
                print("time.time()-a", time.time()-a)

            if (epi + 1) % 50 == 0:
                print('[epoch {}/{}, episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(ep + 1, n_epochs, epi + 1,
                                                                                         n_episodes, ls, ac))
                #print(dimmmm)
    # ????????????
    saver = tf.train.Saver()

    ISOTIMEFORMAT = '_%Y_%m_%d'
    cur_time = time.strftime(ISOTIMEFORMAT, time.localtime())

    file_path = 'model' + cur_time + '_3D'

    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    file_name = file_path + '/model.ckpt'

    saver.save(sess, file_name)


#train_CNN()

dim = 160

def generate_feature_PU():

    # ????????????
    saver = tf.train.Saver()
    ISOTIMEFORMAT = '_%Y_%m_%d'
    cur_time = time.strftime(ISOTIMEFORMAT, time.localtime())

    file_path = 'model_2019_12_08_3D'
    file_name = file_path + '/model.ckpt'

    saver.restore(sess, file_name)

    # ????????????
    f = h5py.File('./PU9_9_100_labeled.h5', 'r')
    test_data = f['data'][:] # (42776, 8100)
    test_label = f['label'][:] # (42776, 9)
    f.close()

    test_data = test_data.reshape(-1, 9, 9, 100)
    #test_data = test_data.transpose((0, 3, 1, 2))

    # ??????????????????????????????????????????
    # S1
    test_data_ = test_data[:1000, :, :, :]
    #test_data_ = test_data_.reshape(-1, n_examples, 100, 9, 9)
    test_data_ = test_data_.reshape(-1, n_examples, 9, 9, 100)

    ll = sess.run(emb_q, feed_dict={q: test_data_})               # ?????????
    print(ll.shape)
    output = ll.reshape(-1, dim)

    # S2
    for i in range(1, 42): # 42776 10988
        temp = test_data[1000 * i:1000 * (i + 1), :, :, :]
        #temp = temp.reshape(-1, n_examples, 100, 9, 9)
        temp = temp.reshape(-1, n_examples, 9, 9, 100)
        ll_temp = sess.run(emb_q, feed_dict={q: temp})
        output_ = ll_temp.reshape(-1, dim)
        output = np.vstack((output, output_))

    # S3
    test_data_ = test_data[-1000:, :, :, :]
    #test_data_ = test_data_.reshape(-1, n_examples, 100, 9, 9)
    test_data_ = test_data_.reshape(-1, n_examples, 9, 9, 100)
    ll = sess.run(emb_q, feed_dict={q: test_data_})
    ll = ll.reshape(-1, dim)
    output_ = ll[-776:, :]
    output = np.vstack((output, output_))

    # ???????????????
    max_value = output.max()
    min_value = output.min()
    output = (output - min_value) / (max_value - min_value)

    print(output.shape) # (42776, 256)
    print(test_label.shape) # (42776, 9)
    # print(np.unique(test_label))

    f = h5py.File('./PU_fea_dim_' + str(dim) + '.h5', 'w')
    f['feature'] = output
    f['label'] = test_label
    f.close()

def generate_feature_IP():

    # ????????????
    saver = tf.train.Saver()
    ISOTIMEFORMAT = '_%Y_%m_%d'
    cur_time = time.strftime(ISOTIMEFORMAT, time.localtime())

    file_path = 'model_2019_12_08_3D'
    file_name = file_path + '/model.ckpt'

    saver.restore(sess, file_name)

    # ????????????
    f = h5py.File('./IP9_9_100_labeled.h5', 'r')
    test_data = f['data'][:] # (9234, 8100)
    test_label = f['label'][:] # (9234, 9)
    f.close()

    test_data = test_data.reshape(-1, 9, 9, 100)
    #test_data = test_data.transpose((0, 3, 1, 2))

    # ??????????????????????????????????????????
    # S1
    test_data_ = test_data[:1000, :, :, :]
    # test_data_ = test_data_.reshape(-1, n_examples, 100, 9, 9)
    test_data_ = test_data_.reshape(-1, n_examples, 9, 9, 100)

    ll = sess.run(emb_q, feed_dict={q: test_data_})  # ?????????
    print(ll.shape)
    output = ll.reshape(-1, dim)

    # S2
    for i in range(1, 10):  # 42776 10988
        temp = test_data[1000 * i:1000 * (i + 1), :, :, :]
        # temp = temp.reshape(-1, n_examples, 100, 9, 9)
        temp = temp.reshape(-1, n_examples, 9, 9, 100)
        ll_temp = sess.run(emb_q, feed_dict={q: temp})
        output_ = ll_temp.reshape(-1, dim)
        output = np.vstack((output, output_))

    # S3
    test_data_ = test_data[-1000:, :, :, :]
    # test_data_ = test_data_.reshape(-1, n_examples, 100, 9, 9)
    test_data_ = test_data_.reshape(-1, n_examples, 9, 9, 100)
    ll = sess.run(emb_q, feed_dict={q: test_data_})
    ll = ll.reshape(-1, dim)
    output_ = ll[-249:, :]
    output = np.vstack((output, output_))

    # ???????????????
    max_value = output.max()
    min_value = output.min()
    output = (output - min_value) / (max_value - min_value)

    print(output.shape) # (42776, 256)
    print(test_label.shape) # (42776, 9)
    # print(np.unique(test_label))

    f = h5py.File('./IP_fea_dim_' + str(dim) + '.h5', 'w')
    f['feature'] = output
    f['label'] = test_label
    f.close()

def generate_feature_SA():

    # ????????????
    saver = tf.train.Saver()
    ISOTIMEFORMAT = '_%Y_%m_%d'
    cur_time = time.strftime(ISOTIMEFORMAT, time.localtime())

    file_path = 'model_2019_12_08_3D'
    file_name = file_path + '/model.ckpt'

    saver.restore(sess, file_name)

    # ????????????
    f = h5py.File('./SA9_9_100_labeled.h5', 'r')
    test_data = f['data'][:] # (42776, 8100)
    test_label = f['label'][:] # (42776, 9)
    f.close()

    test_data = test_data.reshape(-1, 9, 9, 100)
    #test_data = test_data.transpose((0, 3, 1, 2))

    # ??????????????????????????????????????????
    # S1
    test_data_ = test_data[:1000, :, :, :]
    #test_data_ = test_data_.reshape(-1, n_examples, 100, 9, 9)
    test_data_ = test_data_.reshape(-1, n_examples, 9, 9, 100)

    ll = sess.run(emb_q, feed_dict={q: test_data_})               # ?????????
    output = ll.reshape(-1, dim)

    # S2
    for i in range(1, 54): # 42776 10988 19321
        temp = test_data[1000 * i:1000 * (i + 1), :, :, :]
        #temp = temp.reshape(-1, n_examples, 100, 9, 9)
        temp = temp.reshape(-1, n_examples, 9, 9, 100)
        ll_temp = sess.run(emb_q, feed_dict={q: temp})
        output_ = ll_temp.reshape(-1, dim)
        output = np.vstack((output, output_))

    # S3
    test_data_ = test_data[-1000:, :, :, :]
    #test_data_ = test_data_.reshape(-1, n_examples, 100, 9, 9)
    test_data_ = test_data_.reshape(-1, n_examples, 9, 9, 100)
    ll = sess.run(emb_q, feed_dict={q: test_data_})
    ll = ll.reshape(-1, dim)
    output_ = ll[-129:, :]
    output = np.vstack((output, output_))

    # ???????????????
    max_value = output.max()
    min_value = output.min()
    output = (output - min_value) / (max_value - min_value)

    print(output.shape) # (42776, 256)
    print(test_label.shape) # (42776, 9)
    # print(np.unique(test_label))

    f = h5py.File('./SA_fea_dim_' + str(dim) + '.h5', 'w')
    f['feature'] = output
    f['label'] = test_label
    f.close()

def generate_feature_KSC():

    # ????????????
    saver = tf.train.Saver()
    ISOTIMEFORMAT = '_%Y_%m_%d'
    cur_time = time.strftime(ISOTIMEFORMAT, time.localtime())

    file_path = 'model_2019_12_08_3D'
    file_name = file_path + '/model.ckpt'

    saver.restore(sess, file_name)

    # ????????????
    f = h5py.File('./KSC9_9_100_labeled_7-12.h5', 'r')
    test_data = f['data'][:] # (42776, 8100)
    test_label = f['label'][:] # (42776, 9)
    f.close()

    test_data = test_data.reshape(-1, 9, 9, 100)
    #test_data = test_data.transpose((0, 3, 1, 2))

    # ??????????????????????????????????????????
    # S1 3204
    test_data_ = test_data[:1000, :, :, :]
    #test_data_ = test_data_.reshape(-1, n_examples, 100, 9, 9)
    test_data_ = test_data_.reshape(-1, n_examples, 9, 9, 100)

    ll = sess.run(emb_q, feed_dict={q: test_data_})               # ?????????
    output = ll.reshape(-1, dim)

    # S2
    for i in range(1, 3): # 42776 10988 19321
        temp = test_data[1000 * i:1000 * (i + 1), :, :, :]
        #temp = temp.reshape(-1, n_examples, 100, 9, 9)
        temp = temp.reshape(-1, n_examples, 9, 9, 100)
        ll_temp = sess.run(emb_q, feed_dict={q: temp})
        output_ = ll_temp.reshape(-1, dim)
        output = np.vstack((output, output_))

    # S3
    test_data_ = test_data[-1000:, :, :, :]
    #test_data_ = test_data_.reshape(-1, n_examples, 100, 9, 9)
    test_data_ = test_data_.reshape(-1, n_examples, 9, 9, 100)
    ll = sess.run(emb_q, feed_dict={q: test_data_})
    ll = ll.reshape(-1, dim)
    output_ = ll[-204:, :]
    output = np.vstack((output, output_))

    # ???????????????
    max_value = output.max()
    min_value = output.min()
    output = (output - min_value) / (max_value - min_value)

    print(output.shape) # (42776, 256)
    print(test_label.shape) # (42776, 9)
    # print(np.unique(test_label))

    f = h5py.File('./KSC_fea_dim_' + str(dim) + '_7-12.h5', 'w')
    f['feature'] = output
    f['label'] = test_label
    f.close()

def generate_feature_HS15():

    # ????????????
    saver = tf.train.Saver()
    ISOTIMEFORMAT = '_%Y_%m_%d'
    cur_time = time.strftime(ISOTIMEFORMAT, time.localtime())

    file_path = 'model_2019_12_08_3D'
    file_name = file_path + '/model.ckpt'

    saver.restore(sess, file_name)

    # ????????????
    f = h5py.File('./hs9_9_100_labeled.h5', 'r')
    test_data = f['data'][:] # (42776, 8100)
    test_label = f['label'][:] # (42776, 9)
    f.close()

    test_data = test_data.reshape(-1, 9, 9, 100)
    #test_data = test_data.transpose((0, 3, 1, 2))

    # ??????????????????????????????????????????
    # S1
    test_data_ = test_data[:1000, :, :, :]
    #test_data_ = test_data_.reshape(-1, n_examples, 100, 9, 9)
    test_data_ = test_data_.reshape(-1, n_examples, 9, 9, 100)

    ll = sess.run(emb_q, feed_dict={q: test_data_})               # ?????????
    output = ll.reshape(-1, dim)

    # S2
    for i in range(1, 15): # 42776 10988 15029
        temp = test_data[1000 * i:1000 * (i + 1), :, :, :]
        #temp = temp.reshape(-1, n_examples, 100, 9, 9)
        temp = temp.reshape(-1, n_examples, 9, 9, 100)
        ll_temp = sess.run(emb_q, feed_dict={q: temp})
        output_ = ll_temp.reshape(-1, dim)
        output = np.vstack((output, output_))

    # S3
    test_data_ = test_data[-1000:, :, :, :]
    #test_data_ = test_data_.reshape(-1, n_examples, 100, 9, 9)
    test_data_ = test_data_.reshape(-1, n_examples, 9, 9, 100)
    ll = sess.run(emb_q, feed_dict={q: test_data_})
    ll = ll.reshape(-1, dim)
    output_ = ll[-29:, :]
    output = np.vstack((output, output_))

    # ???????????????
    max_value = output.max()
    min_value = output.min()
    output = (output - min_value) / (max_value - min_value)

    print(output.shape) # (42776, 256)
    print(test_label.shape) # (42776, 9)
    # print(np.unique(test_label))

    f = h5py.File('./hs15_fea_dim_' + str(dim) + '.h5', 'w')
    f['feature'] = output
    f['label'] = test_label
    f.close()


def generate_feature_PC():

    # ????????????
    saver = tf.train.Saver()
    ISOTIMEFORMAT = '_%Y_%m_%d'
    cur_time = time.strftime(ISOTIMEFORMAT, time.localtime())

    file_path = 'model_2019_12_08_3D'
    file_name = file_path + '/model.ckpt'

    saver.restore(sess, file_name)

    # ????????????
    f = h5py.File('./PC9_9_100_labeled.h5', 'r')
    test_data = f['data'][:] # (42776, 8100)
    test_label = f['label'][:] # (42776, 9)
    f.close()

    test_data = test_data.reshape(-1, 9, 9, 100)
    #test_data = test_data.transpose((0, 3, 1, 2))

    # ??????????????????????????????????????????
    # S1
    test_data_ = test_data[:1000, :, :, :]
    #test_data_ = test_data_.reshape(-1, n_examples, 100, 9, 9)
    test_data_ = test_data_.reshape(-1, n_examples, 9, 9, 100)

    ll = sess.run(emb_q, feed_dict={q: test_data_})               # ?????????
    print(ll.shape)
    output = ll.reshape(-1, dim)

    # S2
    for i in range(1, 148): # 42776 10988
        temp = test_data[1000 * i:1000 * (i + 1), :, :, :]
        #temp = temp.reshape(-1, n_examples, 100, 9, 9)
        temp = temp.reshape(-1, n_examples, 9, 9, 100)
        ll_temp = sess.run(emb_q, feed_dict={q: temp})
        output_ = ll_temp.reshape(-1, dim)
        output = np.vstack((output, output_))

    # S3
    test_data_ = test_data[-1000:, :, :, :]
    #test_data_ = test_data_.reshape(-1, n_examples, 100, 9, 9)
    test_data_ = test_data_.reshape(-1, n_examples, 9, 9, 100)
    ll = sess.run(emb_q, feed_dict={q: test_data_})
    ll = ll.reshape(-1, dim)
    output_ = ll[-152:, :]
    output = np.vstack((output, output_))

    # ???????????????
    max_value = output.max()
    min_value = output.min()
    output = (output - min_value) / (max_value - min_value)

    print(output.shape) # (42776, 256)
    print(test_label.shape) # (42776, 9)
    # print(np.unique(test_label))

    f = h5py.File('./PC_fea_dim_' + str(dim) + '.h5', 'w')
    f['feature'] = output
    f['label'] = test_label
    f.close()


a = time.time()
generate_feature_PU()
print(time.time()-a)

a = time.time()
generate_feature_IP()
print(time.time()-a)

a = time.time()
generate_feature_SA()
print(time.time()-a)

a = time.time()
generate_feature_PC()
print(time.time()-a)


#generate_feature_KSC()
#generate_feature_HS15()