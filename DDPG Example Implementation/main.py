from QuantConnect.Data.Custom.Tiingo import *


from sediment import WordScore
from env import TradingEnv
from explore import Runner
from agent import TD3

import pandas as pd
import numpy as np

import math

class TwinDelayedDDPG(QCAlgorithm):
    '''Continuous Twin Delayed DDPG model '''

    def Initialize(self):
        
        # settings
        self.FeatureWindow = 10
        self.LookBack = 100 * 2
        self.Test = 20 * 2
        self.LastDataNum = -1
        
        live = False
        
        self.SetStartDate(2019, 1, 1)
        #self.SetEndDate(2019, 10, 31)
        #self.SetBrokerageModel(BrokerageName.Alpaca, AccountType.Cash)
        
        self.symbolDataBySymbol = {}
        self.SymbolList = ['SPY','AAPL','NVDA','AMZN','MSFT','GOOGL','TSLA']#,'MKTX','ABMD','ALGN','AVGO','ULTA','TTWO','FTNT','MA','TGT','TSN']

        self.SecurityList = []
        self.NewsList = []
        
        for symbol in self.SymbolList:
            security = self.AddEquity(symbol, Resolution.Daily)
            self.SecurityList.append(security.Symbol)
            self.symbolDataBySymbol[security.Symbol] = SymbolData(self, security.Symbol, self.FeatureWindow,  Resolution.Daily)
            
        for symbol in self.SecurityList:
            news = self.AddData(TiingoNews, symbol)
            self.NewsList.append(news.Symbol)
            
        self.SetBenchmark("SPY")
        
        self.WS = WordScore()
        
        env = TradingEnv(self.FeatureWindow)
        env.reset()
        
        self.environment = env
        self.observationRun = False
        self.modelIsTraining = False
        
        state_size = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        seed = 0
        
        sizes = (state_size, action_dim, max_action, seed)
        
        self.runnerObj = Runner(self, n_episodes=170, batch_size=5, gamma=0.99, tau=0.005, noise=0.2,\
        noise_clip=0.5, explore_noise=0.1, policy_frequency=2, sizes=sizes)
        
        self.AI_TradeAgent = TD3(self, state_dim=state_size, action_dim=action_dim, max_action=max_action, seed=seed)

        if live:
            self.runnerObj.replay_buffer.load(name='ReplayBuff')
        
        # Set TrainingMethod to be executed immediately
        self.Train(self.TrainingMethod)
        
        # Set TrainingMethod to be executed at 8:00 am every Sunday once a month
        self.Train(self.DateRules.Every(DayOfWeek.Sunday), self.TimeRules.At(6, 0), self.TrainingMethod)
        
    def TrainTimeCheck(self):
        '''Check if new month then Train data'''
        
        today = self.Time
        # can change to month, week, or day for trigger
        weekNum = today.strftime("%V")
        dayNum = today.strftime("%e")
        monthNum = today.strftime("%m")
        
        # trigger logic
        if self.LastDataNum == -1: #self.LastDataNum != monthNum and int(monthNum) % 2 == 0 or 
            self.LastDataNum = monthNum
            # New month time to train TD3!
            return True
        return False
        
    
    def HistoricalData(self, lookBack=100):
        
        historyData = self.History(self.SecurityList, lookBack, Resolution.Daily)
        historyNews = self.History(TiingoNews, self.NewsList, lookBack, Resolution.Daily)
        historyData.dropna(inplace=True)
        historyNews.dropna(inplace=True)
        
        pricesX = {}
        volumeX = {}
        newsX = {}
        
        for symbol in self.SecurityList:
            if not historyData.empty:
                pricesX[symbol.Value] = list(historyData.loc[str(symbol.Value)]['close'])[:-1]
                volumeX[symbol.Value] = list(historyData.loc[str(symbol.Value)]['volume'])[:-1]
            
            # what is the len of data?
            maxValue = len(pricesX[symbol.Value])
                
        for symbol in self.NewsList:
            if not historyNews.empty:
                df = historyNews.loc[symbol].apply(lambda row : self.WS.score(row['description']), axis = 1)
                df.index = pd.to_datetime(df.index, utc=True)
                df = df.resample('1D').sum().fillna(0)
                time = self.Time
                days = pd.date_range(time - timedelta(lookBack), time, freq='D')
                x = np.zeros(lookBack+1)
                x[:] = np.nan
                data = pd.DataFrame({'score':x},index=days)
                data['score'] = df
                data.fillna(0,inplace=True)
                newsX[symbol.Value] = list(data['score'])[-maxValue:]
                
        dictOfDict = {'close':pricesX, 'volume':volumeX, 'score':newsX}
        
        dictOfDf = {k: pd.DataFrame(v) for k,v in dictOfDict.items()}
        
        df = pd.concat(dictOfDf, axis=1)
        
        v1 = pd.Categorical(df.columns.get_level_values(0), 
                           categories=['close', 'volume', 'score'], 
                           ordered=True)
                           
        v2 = pd.Categorical(df.columns.get_level_values(1), 
                            categories=self.SymbolList,
                            ordered=True)
                            
        df.columns = pd.MultiIndex.from_arrays([v2,v1])
        
        return df.sort_index(axis=1, level=[0, 1])
        
    def TrainingMethod(self):
        
        # check to see if we should train the model
        train = self.TrainTimeCheck()
        
        if not train:
            return
        
        # get historical data
        x = self.LookBack
        df = self.HistoricalData(x)
        
        # create envoirments training and testing
        #Env = TradingEnv(obs_len=self.FeatureWindow, df=df)
        trainEnv = TradingEnv(obs_len=self.FeatureWindow, df=df.iloc[:-self.Test])
        testEnv = TradingEnv(obs_len=self.FeatureWindow, df=df.iloc[-self.Test:])
        
        # run observations only once to fill replay buffer with samples
        if not self.observationRun:
            self.runnerObj.observe(trainEnv, 1000)
            self.observationRun = True
        
        # set model to train data
        self.modelIsTraining = True
        self.runnerObj.train(testEnv,testEnv)
        self.modelIsTraining = False
        
    def OnOrderEvent(self, orderEvent):
        self.Debug("{} {}".format(self.Time, orderEvent.ToString()))

    def OnEndOfAlgorithm(self):
        # Save Replay Buffer!
        self.runnerObj.replay_buffer.save(name='ReplayBuff')
        self.Log("{} - TotalPortfolioValue: {}".format(self.Time, self.Portfolio.TotalPortfolioValue))
        self.Log("{} - CashBook: {}".format(self.Time, self.Portfolio.CashBook))

class SymbolData:
    
    def __init__(self, algo, symbol, window, resolution):
        
        self.algo = algo
        self.symbol = symbol
        self.window = window
        self.resolution = resolution
        
        # add each symbol to consolidator
        self.timeConsolidator = TradeBarConsolidator(timedelta(days = 1))
        self.timeConsolidator.DataConsolidated += self.TimeConsolidator
        self.algo.SubscriptionManager.AddConsolidator(symbol, self.timeConsolidator)
        
        # add each symbol to tiingo news
        self.newsAsset = algo.AddData(TiingoNews, symbol)
        
        # temp info used to check changes before trades
        self.weight_temp = 0

        # we will store the historical window here, and keep it a fixed length in update
        self.history_close = []
        self.history_volume = []
        self.history_news = []
        
        # how much of one asset can we buy/short
        self.max_pos = 1 / len(algo.SymbolList)
        self.max_short_pos = 0.0
        
    def update(self, close, volume, symbol):
        '''Update symbols and news with historical data or live data'''
        
        # update history, retain length
        if len(self.history_close)==0:
            hist_df = self.algo.History([self.symbol], timedelta(days=20), self.resolution)
            
            # Case where no data to return for this asset. New asset?
            if 'close' not in hist_df.columns:
                return
            
            hist_df.dropna(inplace=True)
            hist_df.reset_index(level=[0,1],inplace=True)
            hist_df.set_index('time', inplace=True)
            hist_df.dropna(inplace=True)
            
            # store the target time series
            self.history_close = hist_df.close.values[-self.window:]
            self.history_volume = hist_df.volume.values[-self.window:]

        if len(self.history_close) < self.window:
            self.history_close = np.append(self.history_close, close)
            self.history_volume = np.append(self.history_volume, volume)

        else:
            self.history_close = np.append(self.history_close, close)[1:]
            self.history_volume = np.append(self.history_volume, volume)[1:]
        
        if len(self.history_news)==0:
            hist_df = self.algo.History(TiingoNews, self.newsAsset.Symbol, timedelta(days=20), Resolution.Daily)
            hist_df.dropna(inplace=True)
            if not hist_df.empty:
                df = hist_df.loc[self.newsAsset.Symbol].apply(lambda row : self.algo.WS.score(row['description']), axis = 1)
                df.index = pd.to_datetime(df.index, utc=True)
                df = df.resample('1D').sum().fillna(0)
                self.history_news = df.values[-self.window:]
            else:
                self.history_news = np.zeros(self.window)
            
        else:
            hist_df = self.algo.History(TiingoNews, self.newsAsset.Symbol, timedelta(days=1), Resolution.Daily)
            hist_df.dropna(inplace=True)
            if not hist_df.empty:
                df = hist_df.loc[self.newsAsset.Symbol].apply(lambda row : self.algo.WS.score(row['description'] if \
                'description' in row else np.nan), axis = 1)
                
                df.index = pd.to_datetime(df.index, utc=True)
                a = df.resample('1D').sum().fillna(0)
                self.history_news = np.append(self.history_news, np.array(a))[1:]
            else:
                self.history_news = np.append(self.history_news, 0)[1:]
                
    def TimeConsolidator(self, sender, bar):
        '''Live data will be streammed here'''
        
        # return if modle is training
        if self.algo.modelIsTraining:
            self.algo.Debug("Retun, model still training")
            return
        
        # load best AI model
        self.algo.AI_TradeAgent.load("best_avg")
        
        symbol = bar.Symbol
        price = bar.Close
        vol = bar.Volume
        
        # update data arrays with new data
        self.update(price, vol, symbol)
        
        # Check current portfolio for changes
        if self.algo.Securities[symbol].Invested:
            currentweight = (self.algo.Portfolio[symbol].Quantity * price) /self.algo.Portfolio.TotalPortfolioValue
        else:
            currentweight = 0.0
        
        # set ratio currently invested for this symbol
        weight = currentweight
        
        # re-use code in env and get observations with new data
        new_obs = self.algo.environment.next_observation(close_window = self.history_close, \
            volume_window = self.history_volume, news_window=self.history_news)

        # get action from agent
        action = self.algo.AI_TradeAgent.select_action(np.array(new_obs), noise=0)
        
        # View Action
        #self.algo.Debug("{} {} -> {}".format(symbol, new_obs, action))
        
        if action > 0.05:
            
            weight += np.clip(self.algo.environment.normalize(abs(float(action))),0,1) * 0.3
            weight = np.clip(round(weight,4),self.max_short_pos,self.max_pos)
            
            if weight > self.weight_temp:
                self.algo.SetHoldings(symbol, weight, False)
                self.weight_temp = weight
            
        elif action < -0.05:
            
            weight += -(np.clip(self.algo.environment.normalize(abs(float(action))),0,1)) * 0.3
            weight = np.clip(round(weight,4),self.max_short_pos,self.max_pos)
            
            if weight < self.weight_temp:
                self.algo.SetHoldings(symbol, weight, False)
                self.weight_temp = weight
            else:
                pass
        
        else:
            pass