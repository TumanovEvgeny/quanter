#!/usr/bin/python
import numpy as np
import pandas as pd
import sklearn
import glob
from collections import defaultdict
import datetime
from sklearn.preprocessing import LabelEncoder
from copy import copy
from scipy.optimize import *
from sklearn.decomposition import PCA

def DelMinimizer(fun, x0):
    def coef_generator(coef, add):
        args = []
        bul = 0
        if add == -1:
            bul = 1
        for i in range(0, len(coef)):
            if coef[i] == bul:
                coef[i] += add
                args.append(copy(coef))
                coef[i] -= add
        return args
    par_shape = len(x0)
    coef = np.ones(par_shape)
    Q_star = np.inf
    t = 0

    print 'del'
    while sum(coef) > 0:
        print t,
        t = t + 1
        args = coef_generator(coef, -1)
        Qs = map(fun, args)
        argminima = np.argmin(Qs)
        coef = args[argminima]
        #print coef
        Q = Qs[argminima]
        #print Q
        if Q < Q_star:
            t_star = t
            Q_star = Q
        if t - t_star >= 3:
            break
    return {'x' : coef, 'f' : fun(coef)}

def AddDelMinimizer(fun, x0):
    print 'add'
    def coef_generator(coef, add):
        args = []
        bul = 0
        if add == -1:
            bul = 1
        for i in range(0, len(coef)):
            if coef[i] == bul:
                coef[i] += add
                args.append(copy(coef))
                coef[i] -= add
        return args
    par_shape = len(x0)
    coef = np.zeros(par_shape)
    Q_star = np.inf
    t = 0
    while sum(coef) < par_shape:
        print t,
        t = t + 1
        args = coef_generator(coef, 1)
        Qs = map(fun, args)
        argminima = np.argmin(Qs)
        coef = args[argminima]
        #print coef
        Q = Qs[argminima]
        #print Q
        if Q < Q_star:
            t_star = t
            Q_star = Q
        if t - t_star >= 3:
            break
    Q_star = np.inf
    print 'del'
    while sum(coef) > 0:
        print t,
        t = t + 1
        args = coef_generator(coef, -1)
        Qs = map(fun, args)
        argminima = np.argmin(Qs)
        coef = args[argminima]
        #print coef
        Q = Qs[argminima]
        #print Q
        if Q < Q_star:
            t_star = t
            Q_star = Q
        if t - t_star >= 3:
            break
    return {'x' : coef, 'f' : fun(coef)}

def AddDelMinimizer(fun, x0):
    print 'add'
    def coef_generator(coef, add):
        args = []
        bul = 0
        if add == -1:
            bul = 1
        for i in range(0, len(coef)):
            if coef[i] == bul:
                coef[i] += add
                args.append(copy(coef))
                coef[i] -= add
        return args
    par_shape = len(x0)
    coef = np.zeros(par_shape)
    Q_star = np.inf
    t = 0
    while sum(coef) < par_shape:
        print t,
        t = t + 1
        args = coef_generator(coef, 1)
        Qs = map(fun, args)
        argminima = np.argmin(Qs)
        coef = args[argminima]
        #print coef
        Q = Qs[argminima]
        #print Q
        if Q < Q_star:
            t_star = t
            Q_star = Q
        if t - t_star >= 3:
            break
    Q_star = np.inf
    print 'del'
    while sum(coef) > 0:
        print t,
        t = t + 1
        args = coef_generator(coef, -1)
        Qs = map(fun, args)
        argminima = np.argmin(Qs)
        coef = args[argminima]
        #print coef
        Q = Qs[argminima]
        #print Q
        if Q < Q_star:
            t_star = t
            Q_star = Q
        if t - t_star >= 3:
            break
    return {'x' : coef, 'f' : fun(coef)}

def RandomUnifMinimizer(fun, x0, iters = 1000, random_state = 42):
    #np.random.seed(random_state)
    par_shape = len(x0)
    fun_values = [fun(x0)]
    args = [x0]
    for i in range(0, iters):
        if i % round(iters / 20, 0) == 0:
            print str(round(100 * float(i) / iters, 2)) + ' % ',
        vec = np.random.normal(0, 1, par_shape)
        vec = vec / np.sqrt((vec ** 2).sum())
        vec = map(lambda z: max(z, 0), vec)
        fun_values.append(fun(vec))
        args.append(vec)
    argminima = np.argmin(fun_values)
    return {'x' : args[argminima], 'f' : fun_values[argminima]}

class StrategyBaker: 
    def __init__(self, portfolio_weighing, optimize, commiss_per_share, **params):
        # portfolio_weighing in {'best_vs_worst_unif', }
        # optimize in {'no_opt', }
        self.commiss_per_share = commiss_per_share
        self.portfolio_weighing_strategy_params_ = params
        if portfolio_weighing == "best_vs_worst_unif":
            self.portfolio_weighing_strategy = self.best_vs_worst_unif
            
        self.optimize = optimize
    
    
    def data_prepare_(self, features_data, market_data):
        for ticker in features_data.keys():
            if features_data[ticker].shape[0] != market_data[ticker].shape[0]:
                raise ValueError("features data shape doesn't match market data shape on " + ticker)
                
       
        features_data_, market_data_ = defaultdict(), defaultdict()
        
        for key_, key in zip(self.le.transform(features_data.keys()), features_data.keys()):
            features_data_[str(key_)] = features_data[key].copy()
            features_data_[str(key_)].columns = map(lambda z: str(key_) +'_' + z.split('_')[-1], 
                                                              features_data_[str(key_)].columns)
            
        for key_, key in zip(self.le.transform(market_data.keys()), market_data.keys()):
            market_data_[str(key_)] = market_data[key].copy()
            market_data_[str(key_)].columns = map(lambda z: str(key_) +'_' + z.split('_')[-1], 
                                                              market_data_[str(key_)].columns)
       
        features_data_all = features_data_[features_data_.keys()[0]]
        market_data_all = market_data_[features_data_.keys()[0]]
        
        for ticker in features_data_.keys()[1:]:
            features_data_all = pd.concat([features_data_all, features_data_[ticker]], axis = 1)
            market_data_all = pd.concat([market_data_all, market_data_[ticker]], axis = 1)
            
        if self.hedge_data is not None and self.hedge_dict is not None:
            for sym in self.hedge_data.keys():
                self.hedge_data[sym] = self.hedge_data[sym].loc[market_data_all.index].ffill().bfill()
        elif self.hedge_data is not None or self.hedge_dict is not None:
            raise ValueError("if you pass hedge_data you also must pass hedge_dict")

        features_data_all.ffill(inplace = True)
        market_data_all.ffill(inplace = True)
        self.time_index = features_data_all.index
        for ticker in features_data_.keys(): 
            fcols_needed = map(lambda z: ticker + '_' + str(z), range(0, self.features_shape))
            #print fcols_needed
            #print features_data_all.columns
            features_data_[ticker] = np.matrix(features_data_all[fcols_needed])
            mcols_needed = map(lambda z: ticker + z, ["_Open", "_High", "_Low", "_Close"])
            market_data_[ticker] = np.array(market_data_all[mcols_needed])
        
        return features_data_, market_data_
  
        
    def fit(self, features_data_train, market_data_train, hedge_data = None, hedge_dict = None):
        self.hedge_data = hedge_data
        self.hedge_dict = hedge_dict
        self.trading_universe = features_data_train.keys()
        self.le = LabelEncoder()
        self.le.fit(self.trading_universe)
        #rint features_data_train[features_data_train.keys()[0]].shape
        self.features_shape = features_data_train[features_data_train.keys()[0]].shape[1]        
        self.coef_ = np.repeat([(1. / self.features_shape) ** 0.5], self.features_shape)
        #self.coef_ = np.repeat([0], self.features_shape)
        self.features_data_train_, self.market_data_train_ = self.data_prepare_(features_data_train,
                                                                                market_data_train)
        # weights optimize
        def empirical_risk(par):
            pnl = self.backtest_(par)
            return -np.mean(pnl) / np.std(pnl) * np.sqrt(252)
        
        #print empirical_risk(self.coef_)
        #mod = basinhopping(func = empirical_risk, T = 0.25, x0 = self.coef_)
        #mod = minimize(fun = empirical_risk, x0 = self.coef_)

        if self.optimize == 'del':
            mod = DelMinimizer(empirical_risk, x0 = self.coef_)
            print mod
            self.coef_ = mod["x"]
        if self.optimize == 'add_del':
            mod = AddDelMinimizer(empirical_risk, x0 = self.coef_)
            print mod
            self.coef_ = mod["x"]
        if self.optimize == 'bernulli':
            mod = RandomBernulliMinimizer(empirical_risk, x0 = self.coef_)
            print mod
            self.coef_ = mod["x"]
        if self.optimize == 'unif':
            mod = RandomUnifMinimizer(empirical_risk, x0 = self.coef_)
            print mod
            self.coef_ = mod["x"]
        if self.optimize == 'basinhopping':
            pass
            #mod = basinhopping(func = empirical_risk, x0 = self.coef_)
            #print mod
            #self.coef_ = mod["x"]
        if self.optimize == 'scipy_minimize':
            pass
            #mod = basinhopping(fun = empirical_risk, x0 = self.coef_)
            #print mod
            #self.coef_ = mod["x"]


    def alphas_prepare_(self, coef_, features_data):
        indxs_len = features_data['0'].shape[0]
        alphas_ = np.zeros((indxs_len, len(self.trading_universe))) 
        for ticker in map(int, features_data.keys()):
            alpha_vec = features_data[str(ticker)] * np.matrix(coef_).transpose()
            alpha_vec = np.array(alpha_vec)[:, 0]
            alphas_[:, ticker] = alpha_vec #+ coef_[-1]
        return alphas_

    def symbol_pnl_(self, weight, price):   
        shares = weight / price
        shares = np.insert(shares, 0, 0)
        pnl = np.zeros(len(price))
        for i in range(1, len(pnl)):
            pnl[i] = weight[i-1] * (price[i] - price[i-1]) / price[i-1]
            pnl[i] -= np.abs(shares[i] - shares[i-1]) * self.commiss_per_share
            
        return pnl
        
        '''
        shares = map(lambda z: round(z, 0), 100000 * weight / price)
        shares = np.insert(shares, 0, 0)
        pnl = np.zeros(len(price))
        for i in range(1, len(pnl)):
            if np.abs(shares[i] - shares[i - 1]) < 50 and (not ((shares[i] == 0 and shares[i - 1] != 0) or (shares[i] != 0 and shares[i - 1] == 0))):
                shares[i] = shares[i - 1]
            pnl[i] = shares[i] * (price[i] - price[i-1])
            pnl[i] -= min(1, np.abs(shares[i] - shares[i-1]) * self.commiss_per_share)
           
        return pnl
        '''

    def best_vs_worst_unif(self, alphas_, best_n, worst_m):
        need_hedge = self.hedge_data is not None
        if need_hedge:
            self.hedge_weights = defaultdict()
            for sym in self.hedge_data.keys():
                self.hedge_weights[sym] = np.zeros(alphas_.shape[0])

        for row_num, row in enumerate(alphas_):
            nonnans = np.count_nonzero(~np.isnan(row))
            if  nonnans < best_n + worst_m:
                raise ValueError("not NA columns less then best_n + worst_m")

            row[np.isnan(row)] = np.inf
            for i, order in zip(range(0, len(row)), row.argsort().argsort()):
                if order < worst_m:
                    row[i] = -0.5 / worst_m

                elif order >= nonnans - best_n and order < nonnans:
                    row[i] = 0.5 / best_n
                    if need_hedge:
                        self.hedge_weights[self.hedge_dict[self.le.inverse_transform(i)]][row_num] += -0.5 / best_n
                else:
                    row[i] = 0
        return alphas_

    def backtest(self, features_data_test, market_data_test, test_hedge = None):
        raise ValueError('FUNCTION WILL BE DEVELOPED IN FUTURE')
        features, market = self.data_prepare_(features_data_test, market_data_test)
        alphas_ = self.alphas_prepare_(self.coef_, features)
        alphas_ = self.portfolio_weighing_strategy(alphas_, **self.portfolio_weighing_strategy_params_)
        pnls_ = np.zeros(alphas_.shape)
        for ticker in map(int, market.keys()):
            pnls_[:, ticker] = self.symbol_pnl_(alphas_[:, ticker],
                                                           market[str(ticker)][:, 3])
        
        if test_hedge is not None:
            hedge_pl = self.symbol_pnl_(self.hedge_weights, np.array(test_hedge.Close))
            pnls_ = np.concatenate([pnls_, hedge_pl.reshape((len(hedge_pl), 1))], axis = 1)
        sumpnl = np.nansum(pnls_, axis=1)
        print 'sharpe {}'.format(np.mean(sumpnl) / np.std(sumpnl) * np.sqrt(252))
        return sumpnl
        
    def backtest_(self, coef_):
        self.alphas_ = self.alphas_prepare_(coef_, self.features_data_train_)
        self.alphas_ = self.portfolio_weighing_strategy(self.alphas_, **self.portfolio_weighing_strategy_params_)
        
        self.train_pnls_ = np.zeros(self.alphas_.shape)
        for ticker in map(int, self.market_data_train_.keys()):
            self.train_pnls_[:, ticker] = self.symbol_pnl_(self.alphas_[:, ticker],
                                                           self.market_data_train_[str(ticker)][:, 3])
        
        if self.hedge_data is not None:
            for sym in self.hedge_data.keys():
                hedge_pl = self.symbol_pnl_(self.hedge_weights[sym], np.array(self.hedge_data[sym].Close))
                self.hedge_pl = hedge_pl
                self.train_pnls_ = np.concatenate([self.train_pnls_, hedge_pl.reshape((len(hedge_pl), 1))], axis = 1)
        sumpnl = np.nansum(self.train_pnls_, axis=1)
        return sumpnl