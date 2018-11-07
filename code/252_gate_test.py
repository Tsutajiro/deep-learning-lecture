# -*- coding: utf-8 -*-

from lib.perceptron import AND, OR, NAND, XOR

## @file        252_gate_test.py
#  @brief       Chapter 2: Tiny script for verifying logical gates
#  @author      tsutaj
#  @date        November 7, 2018

## @brief       Calculate OR, AND, NAND, XOR result for each binary pattern
def main():
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        resOR   = OR  (xs[0], xs[1])
        resAND  = AND (xs[0], xs[1])
        resNAND = NAND(xs[0], xs[1])
        resXOR  = XOR (xs[0], xs[1])
        print('{0}: OR = {1}, AND = {2}, NAND = {3}, XOR = {4}'.format(xs, resOR, resAND, resNAND, resXOR))

if __name__ == '__main__':
    main()