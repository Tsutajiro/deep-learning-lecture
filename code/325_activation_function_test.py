# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from lib.activation_function import step, sigmoid

## @file        325_activation_function_test.py
#  @brief       Chapter 3: Tiny script for confirming activation functions
#  @author      tsutaj
#  @date        November 7, 2018

## @brief       Show sketches of the graphs of step function and sigmoid function
def main():
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    x1 = np.arange(-5.0, 5.0, 0.1)
    y1 = step(x1)
    ax1.set_title('Step Function')
    ax1.plot(x1, y1)

    ax2 = fig.add_subplot(1, 2, 2)
    x2 = np.arange(-5.0, 5.0, 0.1)
    y2 = sigmoid(x2)
    ax2.set_title('Sigmoid Function')
    ax2.plot(x2, y2)

    plt.show()

if __name__ == '__main__':
    main()