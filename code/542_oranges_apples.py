# -*- coding: utf-8 -*-
## @file        542_oranges_apples.py
#  @brief       Chapter 5: Tiny script of testing back propagation
#  @author      tsutaj
#  @date        November 12, 2018

from lib.layers.MulLayer import MulLayer
from lib.layers.AddLayer import AddLayer

def main():
    # buy some apples and some oranges.
    apple       = 100
    apple_num   = 2
    orange      = 150
    orange_num  = 3
    tax         = 1.1

    # layers
    mul_apple_layer         = MulLayer()
    mul_orange_layer        = MulLayer()
    add_apple_orange_layer  = AddLayer()
    mul_tax_layer           = MulLayer()

    # forward propagation
    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(all_price, tax)

    # backward propagation
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    # print results
    print(price)
    print(dapple_num, dapple, dorange, dorange_num, dtax)

if __name__ == '__main__':
    main()