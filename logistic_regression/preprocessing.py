"""Preprocesing functions for solving toy datasets using logistic regression. """

import numpy as np

def test_case2(X):
    """
    This function implements the required preprocessing to be able to solve
    dataset number 2 of the classification toy datasets using logistic
    regression.
    """
    X_new = X**2
    return X_new

def test_case3(X):
    """
    This function implements the required preprocessing to be able to solve
    dataset number 3 of the classification toy datasets using logistic
    regression.
    """
    return (X[:, 0] * X[:, 1])[:, None]

def test_case4(X):
    """
    This function implements the required preprocessing to be able to solve
    dataset number 4 of the classification toy datasets using logistic
    regression.
    """
    return X

def test_case5(X):
    """
    This function implements the required preprocessing to be able to solve
    dataset number 5 of the classification toy datasets using logistic
    regression.
    """
    return X
