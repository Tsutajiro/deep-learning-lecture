# -*- coding: utf-8 -*-

## @package     MulLayer
#  @brief       Multi layer for back propagation
#  @author      tsutaj
#  @date        November 11, 2018

class MulLayer:
    ## @brief       Define variables which are multiplied
    #  @note        In \p MulLayer, input values for forward propagation are required.
    def __init__(self):
        self.x = None
        self.y = None

    ## @brief       Forward Propagation
    #  @param       x       real number: input argument
    #  @param       y       real number: input argument
    #  @return      a multiplied value: \f$ x \times y \f$
    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    ## @brief       Backward Propagation
    #  @param       x       real number: input argument
    #  @param       y       real number: input argument
    #  @return      a pair of differentiated  values: \f$ \left( \frac{\partial L}{\partial x}, \frac{\partial L}{\partial y} \right) \f$
    #  @note        Let \f$ z = x \times y \f$ be the output of forward propagation by \p MulLayer, and let \f$ L \f$ be the final output of computational graph. We can calculate \f$ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial x} \f$ locally, because \f$ \frac{\partial L}{\partial z} \f$ is given by later layer, and since \f$ z = x \times y \f$, we can easily calculate: \f$ \frac{\partial z}{\partial x} = y \f$. We can also calculate \f$ \frac{\partial L}{\partial y} \f$ locally for the same reason.
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
