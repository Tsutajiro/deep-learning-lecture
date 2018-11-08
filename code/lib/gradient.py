# -*- coding: utf-8 -*-
import numpy as np

## @package     gradient
#  @brief       Chapter 4: Calculate numerical diff and numerical gradient
#  @author      tsutaj
#  @date        November 8, 2018

## @brief       Calculate numerical differentiation
#  @param       func    one-variable function
#  @param       x       real number: the value of variable \p x
#  @return      real number: differential value
def numerical_diff(func, x):
    delta = 1e-4
    return (func(x+delta) - func(x-delta)) / (2*delta)

## @brief       Calculate gradient
#  @param       func    multi-variable function
#  @param       x       an NumPy array or NumPy matrix: the value of variable \p x
#  @return      real NumPy array: gradient
def numerical_gradient(func, x):
    grad = np.zeros_like(x)
    delta = 1e-4

    if x.ndim == 1:
        for idx in range(x.size):
            temp_val = x[idx]
            x[idx] = temp_val + delta
            fx_1 = func(x)

            x[idx] = temp_val - delta
            fx_2 = func(x)

            grad[idx] = (fx_1 - fx_2) / (2*delta)
            x[idx] = temp_val

        return grad
    else:
        for idx, part in enumerate(x):
            grad[idx] = numerical_gradient(func, part)

        return grad

## @brief       Find local minimum value by gradient descent method
#  @param       func        multi-variable function
#  @param       init_x      an NumPy array: initial values
#  @param       lr          \f$\left( 0, 1 \right)\f$ real number: learning rate
#  @param       step_num    integer: the number of repetition
#  @note        Setting learning rate appropriately is very important. If this rate is small, values are rarely updated. If this is large, values diverges to high.
#  @return      real NumPy array: final values which indicate local minimum
def gradient_descent(func, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(func, x)
        x -= lr * grad

    return x
