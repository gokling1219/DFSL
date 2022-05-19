"""Functions for downloading and reading MNIST data."""
from __future__ import print_function
import h5py
import numpy

class DataSet(object):

  def __init__(self, images, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      #assert images.shape[3] == 1
      #images = images.reshape(images.shape[0],images.shape[1] * images.shape[2])
      # Convert from [0, 255] -> [0.0, 1.0].
      #images = images.astype(numpy.float32)
      #images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

class SemiDataSet(object):
    def __init__(self, images, labels, n_labeled):
        self.n_labeled = n_labeled

        # Unlabled DataSet
        self.unlabeled_ds = DataSet(images, labels)

        # Labeled DataSet
        self.num_examples = self.unlabeled_ds.num_examples
        indices = numpy.arange(self.num_examples)
        shuffled_indices = numpy.random.permutation(indices)
        images = images[shuffled_indices]
        labels = labels[shuffled_indices]
        y = numpy.array([numpy.arange(9)[l==1][0] for l in labels])
        n_classes = y.max() + 1
        n_from_each_class = n_labeled / n_classes
        i_labeled = []
        for c in range(n_classes):
            i = indices[y==c][:n_from_each_class]
            i_labeled += list(i)
        l_images = images[i_labeled]
        l_labels = labels[i_labeled]
        self.labeled_ds = DataSet(l_images, l_labels)

    def next_batch(self, batch_size):
        unlabeled_images, _ = self.unlabeled_ds.next_batch(batch_size)
        if batch_size > self.n_labeled:
            labeled_images, labels = self.labeled_ds.next_batch(self.n_labeled)
        else:
            labeled_images, labels = self.labeled_ds.next_batch(batch_size)
        images = numpy.vstack([labeled_images, unlabeled_images])
        return images, labels
        

def read_data_sets(n_labeled = 10):

  class DataSets(object):
    pass

  data_sets = DataSets()
  #f=h5py.File('./data/IP_fea_dim_160.h5','r')
  f = h5py.File('./SA_fea_dim_160.h5', 'r')
  train_images=f['feature'][:]
  #train_images = f['data'][:]
  train_labels=f['label'][:]
  print(train_labels.shape)
  f.close()

  # temp
  # print(train_labels.shape)


  indices = numpy.arange(train_images.shape[0])
  shuffled_indices = numpy.random.permutation(indices)
  images = train_images[shuffled_indices]
  labels = train_labels[shuffled_indices]

  #y = numpy.array([numpy.arange(4)[l==1][0] for l in labels])
  #print(numpy.unique(y))
  y = labels

  n_classes = y.max() + 1
  i_labeled = []


  for c in range(n_classes):

      i = indices[y==c][:n_labeled]
      i_labeled += list(i)
  l_images = images[i_labeled]
  l_labels = labels[i_labeled]
  # l_labels=numpy.argmax(l_labels,1)

  data_sets.train = DataSet(l_images, l_labels)
  #train_labels=numpy.argmax(train_labels,1) 
  data_sets.test = DataSet(train_images, train_labels)

  return data_sets

#read_data_sets(n_labeled = 5)