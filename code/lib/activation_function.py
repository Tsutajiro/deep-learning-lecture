# -*- coding: utf-8 -*-
import numpy as np

## @package     activation_function
#  @brief       Chapter 3: Activation functions
#  @author      tsutaj
#  @date        November 7, 2018

## @brief       Apply step function for all elements of x
#  @param       x an NumPy array
#  @return      an NumPy array created by applying step function to x
def step(x):
    y = x > 0
    return y.astype(np.int)

## @brief       Apply sigmoid funciton for all elements of x
#  @param       x an NumPy array
#  @return      an NumPy array created by applying sigmoid function to x
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

## @brief       Apply ReLU function for all elements of x
#  @param       x an NumPy array
#  @return      an NumPy array created by applying ReLU function to x
def relu(x):
    return np.maximum(0, x)

## @brief       Apply identity function for all elements of x
#  @param       x an NumPy array
#  @return      an NumPy array created by applying identity function to x
def identity(x):
    return x

## @brief       Apply softmax function for all elements of x
#  @param       x an NumPy array
#  @return      an NumPy array created by applying softmax function to x
def softmax(x):
    c = np.max(x) # avoid overflow
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
