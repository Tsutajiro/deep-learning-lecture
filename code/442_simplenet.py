# -*- coding: utf-8 -*-
import sys, os
import numpy as np
from lib.SimpleNet import SimpleNet
from lib.gradient import numerical_gradient

## @file        442_simplenet.py
#  @brief       Chapter 4: Tiny script for testing gradient calculation
#  @author      tsutaj
#  @date        November 8, 2018

def main():
    net = SimpleNet()
    print('initial weight:\n{}'.format(net.W))

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print('prediction: {0}, argmax: {1}'.format(p, np.argmax(p)))

    t = np.array([0, 0, 1])
    print('prediction loss: {}'.format(net.loss(x, t)))

    f = lambda w: net.loss(x, t)
    dW = numerical_gradient(f, net.W)
    print('dW = {}'.format(dW))

if __name__ == '__main__':
    main()