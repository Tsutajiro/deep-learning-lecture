# -*- coding: utf-8 -*-

import urllib.request
import os.path
import gzip
import pickle
import os
import numpy as np

## @package     mnist
#  @brief       Chapter 3: Deal with MNIST dataset
#  @author      tsutaj
#  @date        November 7, 2018

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_image': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_image' : 't10k-images-idx3-ubyte.gz',
    'test_label' : 't10k-labels-idx1-ubyte.gz'
}

# @cond
# abspath   : absolute path
# __file__  : filepath (built-in vaiables)
dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"
# @endcond

train_num   = 60000
test_num    = 10000
image_dim   = (1, 28, 28)
image_size  = 784
class_num   = 10

## @brief       Download objects on Internet
def _download(file_name):
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ", end='')
    # copy objects on Internet into local
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done.")

## @brief       Download images, labels
def download_mnist():
    for v in key_file.values():
        _download(v)

## @brief       Load label files
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ... ", end='')
    # open gzip (mode: read binary)
    with gzip.open(file_path, 'rb') as f:
        # convert buffer to ndarray effectively (offset = 8byte)
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done.")
    
    return labels

## @brief       Load image files
def _load_image(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ... ", end='')
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, image_size)
    print("Done.")

    return data

## @brief       Convert raw dataset into NumPy array
def _convert_numpy():
    dataset = {}
    dataset['train_image'] = _load_image(key_file['train_image'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_image' ] = _load_image(key_file['test_image' ])
    dataset['test_label' ] = _load_label(key_file['test_label' ])

    return dataset

## @brief       Initialize MNIST dataset
def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ... ", end='')
    # write objects using pickle
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done.")

## @brief       Create one-hot label
def _change_one_hot_label(X):
    T = np.zeros((X.size, class_num))
    for idx, row in enumerate(T):
        row[ X[idx] ] = 1

    return T

## @brief       Load MNIST dataset
#  @param       normalize       Normalize array (all values are \f$\left[0, 1\right]\f$) or not (Default: True)
#  @param       flatten         Flatten array (convert into 1-dim) or not (Default: True)
#  @param       one_hot_label   Convert label information into one-hot label (Default: False)
#  @return      ('train_image', 'train_label'), ('test_image', 'test_label')    Tuples which indicates training data and test data
#  @note        If it is the first run, network connection is required for downloading datasets.
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_image', 'test_image'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label' ] = _change_one_hot_label(dataset['test_label' ])

    if not flatten:
        for key in ('train_image', 'test_image'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    # (train: image, label), (test: image, label)
    return (dataset['train_image'], dataset['train_label']), (dataset['test_image'], dataset['test_label'])

if __name__ == '__main__':
    init_mnist()