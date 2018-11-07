# -*- coding: utf-8 -*-
import sys, os
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

## @file        361_load_image.py
#  @brief       Chapter 3: Tiny script for testing MNIST script
#  @author      tsutaj
#  @date        November 7, 2018

## @brief       Show image
def image_show(image):
    pil_image = Image.fromarray(np.uint8(image))
    pil_image.show()

## @brief       Load MNIST dataset, and show sample image
def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

    image = x_train[0]
    label = t_train[0]
    print('label = {}'.format(label))

    print('original shape: {}'.format(image.shape))
    image = image.reshape(28, 28)
    print('reshaped: {}'.format(image.shape))

    image_show(image)

if __name__ == '__main__':
    main()