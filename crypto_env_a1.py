import numpy as np
import pandas as pd
import time
import sys
import random
from sklearn.preprocessing import MinMaxScaler
import talib


class Crypto:
    def __init__(self, name, data_path, start_cash, fee, drawdown_call, fixed_stake, period):
        self.data_path = data_path
        self.start_cash = start_cash
        self.cash = start_cash
        self.fee = fee
        self.drawdown_call = drawdown_call
        self.fixed_stake = fixed_stake
        self.period = period
        self.amt = 0.0
        self.portfolio = start_cash
        self.length = 0

        #9 actions
        self.action_space = ['buy10','buy25','buy50','buy100',
                            'sell10','sell25','sell50','sell100',
                            'hold']
        
        self.n_actions = len(self.action_space)
        self.n_features = period * 9
        self.name = name
        self.step_count = 0
        self.observation = []
        self.done = False
        self.info = ''

        self.current = 0
        self.next = 0
        self.value = 0
        self._build_crypto()

    def _build_crypto(self):
        df = pd.read_csv(self.data_path, usecols=['Timestamp','Open','High','Low','Close','Volume_(Currency)'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.set_index('Timestamp', inplace=True)
        df.interpolate(inplace=True)
        df.columns = ['open','high','low','close','volume']
        #df.drop(df.index[0:self.skip],inplace=True)

        #ema and rsi
        df['rsi'] = talib.RSI(df['close'].values, 14)
        df['ema5'] = talib.SMA(df['close'].values,5)
        df['ema20'] = talib.SMA(df['close'].values,20)
        df['ema60'] = talib.SMA(df['close'].values,60)



        self.data = df.iloc[30000:]
        self.length = len(self.data) - self.period
        del df
    def close(self):
        del self.data

    def reset(self):
        #random episode index
        self.step_count = random.randint(0,self.length-self.period-60)
        #self.step_count = 0
        self.local_step = 0
        self.cash = self.start_cash
        self.amt = 0.0
        self.portfolio = self.start_cash
        self.reward = 0
        self.episode_reward = 0

        #self.observation = np.hstack((self.data[self.step_count:self.step_count+self.period]['close'].values.reshape(-1),self.cash, self.amt))

        self.observation = self.data[self.step_count:self.step_count+self.period].values.reshape(-1)
        #sc = MinMaxScaler()
        #self.observation = sc.fit_transform(self.observation)
        #self.observation = self.observation.reshape(-1)
        #raw_observation = self.data.iloc[self.step_count:self.step_count+self.period]
        #sc = MinMaxScaler()
        #sc_rsi = MinMaxScaler
        #self.observation = sc.fit_transform(raw_observation[raw_observation.columns].values)
        #self.observation = self.observation.reshape(-1)
        self.reward = 0
        self.done = False

        self.length = len(self.data) - self.period

        self.current = self.data.index[self.step_count+self.period-1]
        self.next = self.data.index[self.step_count+self.period]
        self.value = self.data.loc[self.current]['close']
        self.value_next = self.data.loc[self.next]['close']
        self.start_value = self.data.loc[self.current]['close']

        self.info = ''

        return self.observation

    def step(self, action):
        #old_portfolio = self.portfolio
        #diff_value = self.old_value - self.value
        buy_price = self.value * (1. + self.fee)

        if action == 0:   # buy10
            buy_amt = round(self.cash * 0.1 / buy_price, 8)
            if self.cash >= buy_amt * buy_price:
                self.amt += buy_amt
                #self.amt = round(self.amt, 8)
                self.cash -= buy_price * buy_amt
        elif action == 1:   # buy25
            buy_amt = round(self.cash * 0.25 / buy_price, 8)
            if self.cash >= buy_amt * buy_price:
                self.amt += buy_amt
                #self.amt = round(self.amt, 8)
                self.cash -= buy_price * buy_amt
        elif action == 2:   # buy50
            buy_amt = round(self.cash * 0.50 / buy_price, 8)
            if self.cash >= buy_amt * buy_price:
                self.amt += buy_amt
                #self.amt = round(self.amt, 8)
                self.cash -= buy_price * buy_amt
        elif action == 3:   # buy100
            buy_amt = round(self.cash * 1.0 / buy_price, 8)
            if self.cash >= buy_amt * buy_price:
                self.amt += buy_amt
                #self.amt = round(self.amt, 8)
                self.cash -= buy_price * buy_amt
        elif action == 4:   # sell10
            if self.amt > 0 :
                sell_amt = round(self.amt * 0.1 ,8)
                #self.amt -= self.fixed_stake
                self.amt -= sell_amt
                #self.amt = round(self.amt, 8)
                #self.cash += self.value * self.fixed_stake * (1. - self.fee)
                self.cash += self.value * sell_amt * (1. - self.fee)
        elif action == 5:   # sell25
            if self.amt > 0 :
                sell_amt = round(self.amt * 0.25 ,8)
                #self.amt -= self.fixed_stake
                self.amt -= sell_amt
                #self.amt = round(self.amt, 8)
                #self.cash += self.value * self.fixed_stake * (1. - self.fee)
                self.cash += self.value * sell_amt * (1. - self.fee)
        elif action == 6:   # sell50
            if self.amt > 0 :
                sell_amt = round(self.amt * 0.5 ,8)
                #self.amt -= self.fixed_stake
                self.amt -= sell_amt
                #self.amt = round(self.amt, 8)
                #self.cash += self.value * self.fixed_stake * (1. - self.fee)
                self.cash += self.value * sell_amt * (1. - self.fee)
        elif action == 7:   # sell100
            if self.amt > 0 :
                sell_amt = round(self.amt * 1.0 ,8)
                #self.amt -= self.fixed_stake
                self.amt -= sell_amt
                #self.amt = round(self.amt, 8)
                #self.cash += self.value * self.fixed_stake * (1. - self.fee)
                self.cash += self.value * sell_amt * (1. - self.fee)
        elif action == 8:   # hold
            pass

        self.portfolio = self.cash + self.amt * self.value
        portfolio_next = self.cash + self.amt * self.value_next
        #self.reward = float(np.log(self.portfolio/self.start_cash))
        #self.reward = float(np.log(self.portfolio / old_portfolio))
        self.reward = portfolio_next - self.portfolio
        self.portfolio = portfolio_next
        self.episode_reward += self.reward

        #if self.portfolio < self.start_cash * (self.value/self.start_value)* (1 - self.drawdown_call/100.):
        #if self.portfolio < self.start_cash:
        #    if self.reward > 0:
        #        self.reward *= -2.
        #    else:
        #        self.reward *= 2.
            #self.done = True

        if self.local_step == 59:
            self.done = True

        self.step_count += 1
        self.local_step += 1

        self.observation = self.data[self.step_count:self.step_count+self.period].values.reshape(-1)

        #self.observation = np.hstack((self.data.iloc[self.step_count:self.step_count+self.period]['close'].values.reshape(-1),self.cash, self.amt))
        #self.observation = self.data.iloc[self.step_count:self.step_count+self.period]['close'].values.reshape(-1,1)
        #raw_observation = self.data.iloc[self.step_count:self.step_count+self.period]
        #sc = MinMaxScaler()
        #self.observation = sc.fit_transform(raw_observation[raw_observation.columns].values)
        #self.observation = self.observation.reshape(-1)

        self.current = self.data.index[self.step_count+self.period-1]
        self.next = self.data.index[self.step_count+self.period]
        self.value = self.data.loc[self.current]['close']
        self.next_value = self.data.loc[self.next]['close']

        return self.observation, self.reward, self.done

if __name__ == "__main__":
    env = Crypto(name='BTC-USD', data_path='./test.csv', start_cash=1000, fee=0.001, drawdown_call=1, fixed_stake=0.0005, period=240)
    s = env.reset()
    #print(s)
    print(len(s))
    for i in range(64):
        print(env.current)
        s, r, done = env.step(3)
        #print(s_)
        print('%f'%len(s))
        print(r)
        print(done)
        print(env.local_step)
        print(env.portfolio)
        print(env.amt)
        print('----------------------------------')

