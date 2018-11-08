# -*- coding: utf-8 -*-
import numpy as np
from lib.activation_function import softmax
from lib.loss_function import cross_entropy_error
from lib.gradient import numerical_gradient

## @package     SimpleNet
#  @brief       Chapter 4: \f$ 2 \times 3 \f$ shape simple network
#  @author      tsutaj
#  @date        November 8, 2018

class SimpleNet:
    ## @brief       initialize weight randomly using gauss distibution
    def __init__(self):
        # self.W = np.random.randn(2, 3)

        # for verifying
        ## weight parameters
        self.W = np.array([ [0.47355232, 0.9977393, 0.84668094], \
                            [0.85557411, 0.03563661, 0.69422093] ])
 
    ## @brief       Predict by input data and current weight
    #  @param       x       an NumPy array: input data
    #  @return      an NumPy array: prediction result
    def predict(self, x):
        return np.dot(x, self.W)

    ## @brief       Calculate loss between input data \p x and correct label \p t
    #  @param       x       an NumPy array: input data
    #  @param       t       an NumPy array: correct label
    #  @return      real value: loss
    #  @warning     \p t must be one-hot representation.
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t, one_hot_label=True)

        return loss