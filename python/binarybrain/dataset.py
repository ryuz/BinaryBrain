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

def download_mnist(path='.'):
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

