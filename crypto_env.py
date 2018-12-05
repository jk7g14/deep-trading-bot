"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import pandas as pd
import time
import sys
import random
from sklearn.preprocessing import MinMaxScaler

#UNIT = 40   # pixels
#MAZE_H = 4  # grid height
#MAZE_W = 4  # grid width


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

        self.action_space = ['buy','sell','hold']
        self.n_actions = len(self.action_space)
        self.n_features = 180 * 5
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

        self.data = df.iloc[30000:]
        self.length = len(self.data) - self.period
        del df
    def close(self):
        del self.data

    def reset(self):
        #random episode index
        self.step_count = random.randint(0,self.length-self.period-5)
        #self.step_count = 0
        self.local_step = 0
        self.cash = self.start_cash
        self.amt = 0.0
        self.portfolio = self.start_cash
        self.reward = 0
        self.episode_reward = 0

        #self.observation = np.hstack((self.data[self.step_count:self.step_count+self.period]['close'].values.reshape(-1),self.cash, self.amt))

        #self.observation = self.data[self.step_count:self.step_count+self.period]['close'].values.reshape(-1,1)
        #sc = MinMaxScaler()
        #self.observation = sc.fit_transform(self.observation)
        #self.observation = self.observation.reshape(-1)
        raw_observation = self.data.iloc[self.step_count:self.step_count+self.period]
        sc = MinMaxScaler()
        self.observation = sc.fit_transform(raw_observation[raw_observation.columns].values)
        self.observation = self.observation.reshape(-1)
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

        if action == 0:   # buy
            #if self.cash >= self.value * self.fixed_stake * (1. + self.fee):
            self.amt += self.fixed_stake
            self.amt = round(self.amt, 3)
            self.cash -= self.value * self.fixed_stake *  (1. + self.fee)
        elif action == 1:   # sell
            #if self.amt > 0 :
                #sell_amt = self.amt * 0.5
            self.amt -= self.fixed_stake
            self.amt = round(self.amt, 3)
            self.cash += self.value * self.fixed_stake * (1. - self.fee)
        elif action == 2:   # hold
            pass

        self.portfolio = self.cash + self.amt * self.value
        portfolio_next = self.cash + self.amt * self.value_next
        #self.reward = float(np.log(self.portfolio/self.start_cash))
        #self.reward = float(np.log(self.portfolio / old_portfolio))
        self.reward = portfolio_next - self.portfolio
        self.portfolio = portfolio_next
        self.episode_reward += self.reward

        if self.portfolio < self.start_cash * (self.value/self.start_value)* (1 - self.drawdown_call/100.):
            self.reward *= 2
            #self.done = True

        if self.local_step == 63:
            self.done = True

        self.step_count += 1
        self.local_step += 1

        #self.observation = self.data[self.step_count:self.step_count+self.period].values.reshape(-1)

        #self.observation = np.hstack((self.data.iloc[self.step_count:self.step_count+self.period]['close'].values.reshape(-1),self.cash, self.amt))
        #self.observation = self.data.iloc[self.step_count:self.step_count+self.period]['close'].values.reshape(-1,1)
        raw_observation = self.data.iloc[self.step_count:self.step_count+self.period]
        sc = MinMaxScaler()
        self.observation = sc.fit_transform(raw_observation[raw_observation.columns].values)
        self.observation = self.observation.reshape(-1)

        self.current = self.data.index[self.step_count+self.period-1]
        self.next = self.data.index[self.step_count+self.period]
        self.value = self.data.loc[self.current]['close']
        self.next_value = self.data.loc[self.next]['close']

        return self.observation, self.reward, self.done

if __name__ == "__main__":
    env = Crypto(name='BTC-USD', data_path='./test.csv', start_cash=1000, fee=0.001, drawdown_call=1, fixed_stake=0.001, period=180)
    s = env.reset()
    #print(s)
    print(len(s))
    for i in range(64):
        print(env.current)
        s_, r, done = env.step(0)
        #print(s_)
        print('%f'%len(s))
        print(r)
        print(done)
        print(env.local_step)
        print(env.portfolio)
        print('----------------------------------')

