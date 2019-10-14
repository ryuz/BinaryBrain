# coding: utf-8

import os
import urllib.request
import tarfile
import gzip
import shutil
import pickle
import numpy as np


dataset_path = os.path.join(os.path.expanduser('~'), '.binarybrain', 'dataset')
# mnsit_pickle = 'mnist.pickle'

def wget(url, filename):
    print(url)
    with urllib.request.urlopen(url) as r:
        with open(filename, 'wb') as f:
            f.write(r.read())

def tar_extractall(tar_filename, extract_path):
    with tarfile.open(tar_filename, 'r') as f_tar:
        f_tar.extractall(extract_path)

def gzip_extractall(gz_filename, ext_filename):
    with gzip.open(gz_filename, 'rb') as f_gz:
        with open(ext_filename, 'wb') as f_ext:
            shutil.copyfileobj(f_gz, f_ext)

def gzip_download_and_extract(url, gz_filename, ext_filename):
    if not os.path.exists(ext_filename):
        if not os.path.exists(gz_filename):
            print('dwonload %s from %s' % (gz_filename, url))
            wget(url, gz_filename)
        gzip_extractall(gz_filename, ext_filename)

def download_mnist(path=''):
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    names = [('train-images-idx3-ubyte.gz', 'train-images-idx3-ubyte'),
             ('train-labels-idx1-ubyte.gz', 'train-labels-idx1-ubyte'),
             ('t10k-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte'),
             ('t10k-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte'),] 
    for name in names:
        url = base_url + name[0]
        gz_filename = os.path.join(path, name[0])
        ext_filename = os.path.join(path, name[1])
        gzip_download_and_extract(url, gz_filename, ext_filename)

def read_mnist_image_file(file_name):
    with open(file_name, 'rb') as f:
        _   = np.fromfile(f, np.uint8, 16) # header
        img = np.fromfile(f, np.uint8, -1) # data
    img = img.astype(np.float32)
    img /= 255.0
    img = img.reshape(-1, 28*28*1)
    return img

def read_mnist_label_file(file_name):
    with open(file_name, 'rb') as f:
        _ = np.fromfile(f, np.uint8, 8)  # header
        l = np.fromfile(f, np.uint8, -1) # data
    labels = np.zeros((len(l), 10), np.float32)
    for i, j in enumerate(l):
        labels[i][j] = 1.0
    return labels

def read_mnist(path=''):
    td = {}
    td['x_train'] = read_mnist_image_file(os.path.join(dataset_path, 'train-images-idx3-ubyte'))
    td['t_train'] = read_mnist_label_file(os.path.join(dataset_path, 'train-labels-idx1-ubyte'))
    td['x_test']  = read_mnist_image_file(os.path.join(dataset_path, 't10k-images-idx3-ubyte'))
    td['t_test']  = read_mnist_label_file(os.path.join(dataset_path, 't10k-labels-idx1-ubyte'))
    td['x_shape'] = [28, 28, 1]
    td['t_shape'] = [10]
    return td

def load_mnist():
    os.makedirs(dataset_path, exist_ok=True)
    download_mnist(path=dataset_path)
    td = read_mnist(path=dataset_path)
    return td


def download_cifar10(path=''):
    tgz_filename = os.path.join(path, 'cifar-10-python.tar.gz')
    if not os.path.exists(tgz_filename):
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        print('dwonload %s from %s' % (tgz_filename, url))
        wget(url, tgz_filename)
    if not os.path.exists(os.path.join(path, 'cifar-10-batches-py', 'data_batch_1'))  \
            or not os.path.exists(os.path.join(path, 'cifar-10-batches-py', 'data_batch_2'))  \
            or not os.path.exists(os.path.join(path, 'cifar-10-batches-py', 'data_batch_3'))  \
            or not os.path.exists(os.path.join(path, 'cifar-10-batches-py', 'data_batch_4'))  \
            or not os.path.exists(os.path.join(path, 'cifar-10-batches-py', 'data_batch_5'))  \
            or not os.path.exists(os.path.join(path, 'cifar-10-batches-py', 'test_batch')):
        tar_extractall(tgz_filename, path)

def read_cifar10_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

def read_cifar10(path=''):
    data_batch_1 = read_cifar10_pickle(os.path.join(path, 'cifar-10-batches-py', 'data_batch_1'))
    data_batch_2 = read_cifar10_pickle(os.path.join(path, 'cifar-10-batches-py', 'data_batch_2'))
    data_batch_3 = read_cifar10_pickle(os.path.join(path, 'cifar-10-batches-py', 'data_batch_3'))
    data_batch_4 = read_cifar10_pickle(os.path.join(path, 'cifar-10-batches-py', 'data_batch_4'))
    data_batch_5 = read_cifar10_pickle(os.path.join(path, 'cifar-10-batches-py', 'data_batch_5'))
    test_batch   = read_cifar10_pickle(os.path.join(path, 'cifar-10-batches-py', 'test_batch'))

    x_train = np.vstack((data_batch_1[b'data'], data_batch_2[b'data'], data_batch_3[b'data'], data_batch_4[b'data'], data_batch_5[b'data']))
    l_train = data_batch_1[b'labels'] + data_batch_2[b'labels'] + data_batch_3[b'labels'] + data_batch_4[b'labels'] + data_batch_5[b'labels']
    x_test  = test_batch[b'data']
    l_test  = test_batch[b'labels']

    t_train = np.zeros((len(l_train), 10), np.float32)
    for i, l in enumerate(l_train):
        t_train[i][l] = 1.0
    
    t_test = np.zeros((len(l_test), 10), np.float32)
    for i, l in enumerate(l_test):
        t_test[i][l] = 1.0

    td = {}
    td['x_train'] = np.array(x_train).astype(np.float32) / 255.0
    td['x_test']  = np.array(x_test).astype(np.float32)  / 255.0
    td['x_shape'] = [32, 32, 3]
    td['t_train'] = t_train
    td['t_test']  = t_test
    td['t_shape'] = [10]
    return td

def load_cifar10():
    os.makedirs(dataset_path, exist_ok=True)
    download_cifar10(path=dataset_path)
    td = read_cifar10(path=dataset_path)
    return td
