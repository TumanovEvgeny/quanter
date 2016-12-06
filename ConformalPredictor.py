#!/usr/bin/python

from sklearn.utils import check_array
import numpy as np
from joblib import Parallel, delayed

class ConformalPredictor:
    '''
    Common usage:
    
    X_train = np.array([-1, -0.75, -0.5, -0.25, 0.25,0.75, 1])
    X_train = X_train.reshape((len(X_train), 1))
    X_train = np.concatenate([X_train, X_train ** 2], axis = 1)
    y_train = np.array([1, 0.75, 0.5, 0.25, 0.25, 0.75, 1.0])
    print X_train
    cp = ConformalPredictor(LinearRegression)
    cp.fit(X_train, y_train)

    X_test = np.array([[0, 0]])
    cp.predict_conf_interval(X_test, 0.95, 0.001, False)
    
    
    
    '''
    def __init__(self, regressor, **params):
        '''
        pass here a sklearn regression model with params
        call example:
        from sklearn.linear_model import LinearRegression
        cp = ConformalPredictor(LinearRegression, normalize = True, n_jobs = -1)
        cp.model
        
        '''
        self.regressor = regressor(**params)
        self.model = regressor(**params)
        
    def fit(self, X_train, y_train):
        '''
        call sklearn like
        X_train : numpy array  (n_samples, n_features)
           Training data
        y_train : numpy array of shape (n_samples, 1)
           Target values
        
        '''
        X_train = check_array(X_train, accept_sparse=['csr', 'csc', 'coo'])
        self.X_train = X_train
        self.y_train = y_train
        self.n_samples, self.n_features = X_train.shape
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        '''
        predict just call predict of passed model
        
        call sklearn like
        X_test :  numpy array of shape (n_samples, n_features)
            Test data
        
        '''
        
        return self.model.predict(X_test)
        
    def _m(self, x, y):
        x_ext = np.concatenate([self.X_train, x], axis = 0)
        y_ext = np.append(self.y_train, y)
        _conformal_model = self.regressor
        _conformal_model.fit(x_ext, y_ext)
        y_hat_ext = _conformal_model.predict(x_ext)

        eps = (y_hat_ext - y_ext) ** 2
        
        return float(eps.argsort().argsort()[-1]) + 1
    
    def _predict_conf_interval_one_sample(self, x_test, conf, bin_sizer):
        x_test = x_test.reshape((1, self.n_features))
        y_hat = self.model.predict(x_test)[0]
        y_range = (np.max(self.y_train) - np.min(self.y_train)) / 2
      
        y_step = y_range * bin_sizer
        
        left, right = y_hat - y_range, y_hat + y_range
        
        y_interval = np.arange(left, right, y_step)
        fun = lambda x, y: self._m(x, y) / (self.n_samples + 1) <= conf
        # DO PARALLEL
        #ins = Parallel(n_jobs = 2)(delayed(fun)(x_test, y) for y in y_interval)     
        #y_conf_interval = [y for y, flag in zip(y_interval, ins) if flag]
        
        y_conf_interval = [y for y in y_interval if fun(x_test, y)]
        return np.array([np.min(y_conf_interval), np.max(y_conf_interval)])
                
    
    def predict_conf_interval(self, X_test, conf, bin_sizer):
        '''
        Returns confidence intervals for all samples in X_test
        
        call sklearn like
        X_test : numpy array of shape (n_samples, n_features)
            Test data
        bin_sizer : bin_size =  bin_sizer * (range of y)
        conf : confidence level
        '''
        X_test = check_array(X_test, accept_sparse=['csr', 'csc', 'coo'])   
        conf_intervals = np.zeros((X_test.shape[0], 2))
        for x, i in zip(X_test, range(conf_intervals.shape[0])):
            conf_intervals[i] = self._predict_conf_interval_one_sample(x, conf, bin_sizer)
        
        return conf_intervals