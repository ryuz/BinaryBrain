# coding: utf-8

import os
import urllib.request
import tarfile
import gzip
import shutil


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
            wget(url, gz_filename)
        gzip_extractall(gz_filename, ext_filename)

def download_mnist(download_path='.'):
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    names = [('train-images-idx3-ubyte.gz', 'train-images-idx3-ubyte'),
             ('train-labels-idx1-ubyte.gz', 'train-labels-idx1-ubyte'),
             ('t10k-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte'),
             ('t10k-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte'),] 
    for name in names:
        url = base_url + name[0]
        gz_filename = os.path.join(download_path, name[0])
        ext_filename = os.path.join(download_path, name[1])
        gzip_download_and_extract(url, gz_filename, ext_filename)



