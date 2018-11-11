# -*- coding: utf-8 -*-

## @package     AddLayer
#  @brief       Add layer for back propagation
#  @author      tsutaj
#  @date        November 11, 2018

class AddLayer:
    ## @brief       Do nothing
    #  @note        In \p AddLayer, we do not need to remember any value.
    def __init__(self):
        pass

    ## @brief       Forward Propagation
    #  @param       x       real number: input argument
    #  @param       y       real number: input argument
    #  @return      an added value: \f$ x + y \f$
    def forward(self, x, y):
        return x + y

    ## @brief       Backward Propagation
    #  @param       x       real number: input argument
    #  @param       y       real number: input argument
    #  @return      a pair of differentiated  values: \f$ \left( \frac{\partial L}{\partial x}, \frac{\partial L}{\partial y} \right) \f$
    #  @note        Let \f$ z = x + y \f$ be the output of \p AddLayer. Since \f$ \frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} = 1 \f$, all we have to do is passing \p dout.
    def backward(self, dout):
        dx, dy = dout, dout
        return dx, dy