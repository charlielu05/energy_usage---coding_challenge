# -*- coding: utf-8 -*-
""" Coding challenge imports
Date: 15/07/2018
Author: Charlie Lu
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from xgboost import plot_importance


def my_plot_importance(booster, figsize, **kwargs): 
    """
    Plots feature importance from XGBoost regressor. 
    from: https://stackoverflow.com/questions/40081888/xgboost-plot-importance-figure-size
    Parameters
    ----------
    booster : object
        XGBoost object
    figsize : tuple
        size of plot
    """
    _, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax, **kwargs)

def prediction_plot(y_true, y_predict):
    """
    Function used to plot prediction vs true and also R-squared value.
    This was replaced by prediction plot from yellowbrick package

    Parameters
    ----------
    y_true : pandas dataframe
        true prediction values
    y_predict : pandas dataframe
        predicted values
    Returns
    -------
    R-squared score : string
        The R-squared value 
    """
    _ , ax = plt.subplots()
    ax.scatter(y_true, y_predict, edgecolors=(0, 0, 0))
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    print('R-squared: {}'.format(r2_score(y_true, y_predict)))

def nmse_cv(model, X, y, n_folds = 5):
    """
    Calculates the mean of negative MSE from K-fold cross-validation.

    Parameters
    ----------
    model : object
        the model we want to use, Linear Regression, XGBoost...etc
    X : pandas dataframe
        predictor values 
    y : pandas dataframe
        prediction values
    n_folds : int
        number of folds we want to split the data into
    Returns
    -------
    NMSE score : string
        The mean/average NMSE score from cross-validation
    """
    score = cross_val_score(model, X, y, cv= n_folds, scoring = 'neg_mean_squared_error').mean()

    print('Mean NMSE: {}'.format(score))


class CV(object):
    def __init__(self, strategy, clf, params = None):
        """
        Creates a Cross-Validation class to test different imputation methods and classifiers
        This was later replaced with rolling window mean to impute missing values, rendering the
        class with imputation pipeline redundant.

        Parameters
        ----------
        strategy : string
            strategy used for imputation, valid options: mean, median, mode
        clf : object
            the model we want to use, Linear Regression, XGBoost...etc
        params : dict
            the arguments used to construct the model
        Returns
        -------
        dataset
            The ``Dataset`` abstraction
        intercept_decay
            The intercept decay
        """
        self.strategy = strategy
        self.estimator = clf(**params)
        self.pipeline = self.create_pipeline(self.strategy, self.estimator )
        self.score = None
        self.prediction = None

    def create_pipeline(self, strategy = None, estimator = None):
        """
        Creates the pipeline.

        Parameters
        ----------
        strategy : string
            strategy used for imputation, valid options: mean, median, mode
        estimator : object
            the model we want to use, Linear Regression, XGBoost...etc
        Returns
        -------
        pipeline : object
            returns a pipeline that we can use to streamline imputation and model fitting
        """
        pipeline = Pipeline([('imputer', Imputer(missing_values = 'NaN', strategy = self.strategy,
        axis = 0)), ('model', self.estimator)])

        return pipeline
    
    def mean_score(self, x, y_true):
        """
        Returns the mean/average cross-validation score.

        Parameters
        ----------
        x : pandas dataframe
            predictor values 
        y : pandas dataframe
            prediction values
        Returns
        -------
        score : int
            returns the mean/average score from 10 fold cross validation
        """
        self.score = cross_val_score(self.pipeline, x, y_true, cv=10).mean()
        
        return (self.score)
    
    def predict(self, x, y_true):
        """
        Returns the cross-validated predictions.

        Parameters
        ----------
        x : pandas dataframe
            predictor values 
        y : pandas dataframe
            prediction values
        Returns
        -------
        prediction : ndarray
            returns the cross-validated prediction from 10 fold cross validation
        """
        self.prediction = cross_val_predict(self.pipeline, x, y_true, cv= 10)

        return (self.prediction)
