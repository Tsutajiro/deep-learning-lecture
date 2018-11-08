# -*- coding: utf-8 -*-
import numpy as np

## @package     loss_function
#  @brief       Chapter 4: Functions which calculate train loss
#  @author      tsutaj
#  @date        November 8, 2018

## @brief       Calculate mean squared error between x and y
#  @brief       function: \f$ E = \frac{1}{2} \sum_{k} (y_k - t_k)^2 \f$
#  @param       y       an NumPy array output by neural network
#  @param       t       an NumPy array representing the correct label
#  @param       one_hot_label   boolean if \p t is one-hot representation or not  
#  @warning     It is not verified when \p t is not one-hot representation!
#  @note        This function works both single data input and batch input.
def mean_squared_error(y, t, one_hot_label=False):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    if one_hot_label == False:
        # convert to one-hot representation
        num_of_labels = y.shape[1]
        t = np.eye(num_of_labels)[t]
    return 0.5 * np.sum((y - t) ** 2) / batch_size

## @brief       Calculate cross entropy error between x and y
#  @brief       function: \f$ E = - \sum_{k} t_k \log y_k \f$
#  @param       y               an NumPy array output by neural network
#  @param       t               an NumPy array representing the correct label
#  @param       one_hot_label   boolean if \p t is one-hot representation or not  
#  @warning     It is not verified when \p t is not one-hot representation!
#  @note        This function works both single data input and batch input.
def cross_entropy_error(y, t, one_hot_label=False):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7

    if one_hot_label == True:
        return -np.sum(t * np.log(y + delta)) / batch_size
    else:
        return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size