# -*- coding: utf-8 -*-
import numpy as np

## @package     perceptron
#  @brief       Basic digital logic gates (Chapter 2)
#  @author      tsutaj
#  @date        November 6, 2018

## @brief       Calculate \f$x_1\f$ \f$\mathrm{AND}\f$ \f$x_2\f$
#  @param       x1 input (boolean)
#  @param       x2 input (boolean)
#  @return      boolean \f$x_1\f$ \f$\mathrm{AND}\f$ \f$x_2\f$
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    if np.sum(w * x) + b <= 0:
        return 0
    else:
        return 1

## @brief       Calculate \f$x_1\f$ \f$\mathrm{OR}\f$ \f$x_2\f$
#  @param       x1 input (boolean)
#  @param       x2 input (boolean)
#  @return      boolean \f$x_1\f$ \f$\mathrm{OR}\f$ \f$x_2\f$
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    if np.sum(w * x) + b <= 0:
        return 0
    else:
        return 1

## @brief       Calculate \f$x_1\f$ \f$\mathrm{NAND}\f$ \f$x_2\f$
#  @param       x1 input (boolean)
#  @param       x2 input (boolean)
#  @return      boolean \f$x_1\f$ \f$\mathrm{NAND}\f$ \f$x_2\f$
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    if np.sum(w * x) + b <= 0:
        return 0
    else:
        return 1

## @brief       Calculate \f$x_1\f$ \f$\mathrm{XOR}\f$ \f$x_2\f$
#  @param       x1 input (boolean)
#  @param       x2 input (boolean)
#  @return      boolean \f$x_1\f$ \f$\mathrm{XOR}\f$ \f$x_2\f$
#  @note        XOR gate takes non-linear region, so you cannot describe it using only single perceptron. The combination of AND / OR / NAND gate enable you to describe this gate (2-layered perceptron).
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)