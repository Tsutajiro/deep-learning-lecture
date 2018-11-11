# -*- coding: utf-8 -*-
## @package     Sigmoid
#  @brief       Sigmoid layer for back propagation
#  @author      tsutaj
#  @date        November 11, 2018

class Sigmoid:
    ## @brief       Define a value for remembering the output of forward propagation
    #  @note        This value is used in back propagation.
    def __init__(self):
        self.out = None

    ## @brief       Forward Propagation
    #  @param       x       a real number: input
    #  @return      an real number: sigmoid function output
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    ## @brief       Back Propagation
    #  @param       dout    a real number: input
    #  @return      a real number: differentiated value
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx