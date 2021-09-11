import numpy as np
import pandas as pd
from collections import deque
import random
from gym import spaces
import math
from scipy.stats import linregress
import random

class TradingEnv(object):
    
    def __init__(self, obs_len=10, df=None):
        
        self.window = obs_len
        self.data = df
        
        if df is None:
            self.data = self.dummy_data()
            
        x = list(self.data.columns.get_level_values(0))
        x = list(dict.fromkeys(x))
        
        self.SymbolList = x
        self.CountIter = -1
        self.MaxCount = len(x)
        self.action_space = spaces.Box(-1, +1, (1,), dtype=np.float32)
        
    def dummy_data(self):
        x1 = np.zeros(100)
        close = {'symbol_1':x1,'symbol_2': x1,'symbol_3':x1}
        vol = {'symbol_1':x1,'symbol_2': x1,'symbol_3':x1}
        score = {'symbol_1':x1,'symbol_2': x1,'symbol_3':x1}
        y = {'close':close, 'volume':vol, 'score':score}
        dict_of_df = {k: pd.DataFrame(v) for k,v in y.items()}
        df = pd.concat(dict_of_df, axis=1)
        v = pd.Categorical(df.columns.get_level_values(0), 
                           categories=['close', 'volume', 'score'], 
                           ordered=True)
        v2 = pd.Categorical(df.columns.get_level_values(1), 
                            categories=['symbol_1', 'symbol_2', 'symbol_3'],
                            ordered=True)
        df.columns = pd.MultiIndex.from_arrays([v2,v])
        return df.sort_index(axis=1, level=[0, 1])
        
    def reset(self, randomIndex=False):
        """reset and set data to zero or null"""
        
        # randomly pick stock if true
        if randomIndex:
            self.CountIter = random.randint(0, int(self.MaxCount))
        
        # iterate through list to train
        if self.CountIter + 1 >= self.MaxCount:
            self.CountIter = -1
        
        self.CountIter += 1
        
        # used to identify what symbol data is used
        self.sym = self.SymbolList[self.CountIter]
        
        df = self.data[self.sym]
        
        self.close = df['close'].values
        self.volume = df['volume'].values
        self.news = df['score'].values
        self.returns = df['close'].pct_change().values
        
        # start index so rolling window is full of data
        self.ts_index = self.window + 1

        # get obs
        c_window, v_window, n_window = self.on_data()
        observations = self.next_observation(close_window=c_window, volume_window=v_window, news_window=n_window)
        
        # get obs space
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(observations),), dtype=np.float32)
        
        # used to calc strategy perf
        self.strat_returns = []
        
        return observations
        
    def std(self,x):
        '''generate standard deviation'''
        y = (x - x.mean())/x.std()
        return y[-1]
        
    def exponential_regression(self, data):
        log_c = np.log(data)
        x = np.arange(len(log_c))
        slope, _, rvalue, _, _ = linregress(x, log_c)
        return (1 + slope) * (rvalue ** 2)
        
    def regression(self, data):
        x = np.arange(len(data))
        slope, _, rvalue, _, _ = linregress(x, data)
        return (1 + slope) * (rvalue ** 2) 
    
    def next_observation(self, close_window, volume_window, news_window):
        '''return observations'''
        
        # calculate log of closing prices (feature 1)
        sed = news_window.mean() / 100
        
        # log diff of price
        #diff_c = np.diff(np.log(close_window)).sum()*100
        
        # log diff of volume
        #diff_v = np.diff(np.log(volume_window)).sum()*5
        
        # volitility of price
        #std = np.log(close_window).std()
        
        # how well the exp slope is correlated with itself 
        #exp_reg = self.exponential_regression(close_window)
        
        # how well the slope is correlated with itself 
        lin_reg = self.regression(close_window)
        
        # last std value of close
        col = self.std(close_window)
        
        # last std value of volume
        vol = self.std(volume_window)
        
        # last std value of sediment
        seddir = self.std(news_window)
        
        # combine mutliple features
        obs = np.concatenate(([sed],[col],[lin_reg],[vol],[seddir]), axis=0)
        
        # if nan replace nans with zero
        where_are_NaNs = np.isnan(obs)
        obs[where_are_NaNs] = 0
        
        return obs

    def on_data(self):
        '''update data'''
        # where are we in index?
        step = self.ts_index
        
        close_window = self.close[step-self.window:step]
        volume_window = self.volume[step-self.window:step]
        news_window = self.volume[step-self.window:step]
        
        return close_window, volume_window, news_window
        
        
    def get_reward(self, trade=0):
        '''rewards for trading performance'''
        
        # where are we in index?
        step = self.ts_index
        
        # calculate reward 
        reward = self.returns[step] * trade
        
        self.strat_returns.append(reward)
        
        return reward if np.isfinite(reward) else 0
        

    def normalize(self,x):
        ''' 
        greater than 0.15 == buy scaled 0-1
        in between +0.15 & -0.15 == do nothing
        less than -0.15 == sell scaled 0-1
        '''
        return np.round((1/0.95*x)-0.05264,3)
        
    def step(self, action):
        '''step through envoirment'''
        
        done = False
        
        # get action from neural network
        action = float(action[0])
        
        # buy sell do nothing logic
        if action >= 0.05:
            # bet size buy
            size = np.clip(self.normalize(abs(action)),0,1)
            
        elif action <= -0.05:
            # bet size sell
            size = -(np.clip(self.normalize(abs(action)),0,1))
        
        else:
            # do nothing
            size = 0

        # if done, break and return final values
        if self.ts_index + 2 >= len(self.close):

            done = True
            
            # get reward
            reward = self.get_reward(trade=size)
            
            # step through next iteration of data
            c_window, v_window, n_window = self.on_data()
            
            # gets obs
            observations = self.next_observation(close_window=c_window, volume_window=v_window, news_window=n_window)
            
            return observations, reward, done, self.ts_index
            
        # get reward
        reward = self.get_reward(trade=size)
        
        # add one to timestep index (different from current index)
        self.ts_index += 1

        # step through next iteration of data
        c_window, v_window, n_window = self.on_data()
        
        # get features
        observations = self.next_observation(close_window=c_window, volume_window=v_window, news_window=n_window)
        
        return observations, reward, done, self.ts_index