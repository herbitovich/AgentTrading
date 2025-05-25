import gym
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from gym import spaces
import torch

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, trend_predictor, predict_steps=5, initial_balance=10000, penalty_coef=0.001):
        super(StockTradingEnv, self).__init__()

        self.trend_predictor = trend_predictor
        if self.trend_predictor: 
            self.trend_predictor.eval()
        self.predict_steps = predict_steps
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.penalty_coef = penalty_coef

        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([3, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(3,), 
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.prev_value = self.initial_balance
        
        return self._next_observation()

    def get_observation(self):
        df = self.df[self.current_step:]
        if self.trend_predictor:
            found = False
            for window in df.rolling(window=self.trend_predictor.window_size):
                if len(window)==self.trend_predictor.window_size:
                    window = window.to_list()
                    found = True
                    scaler = MinMaxScaler()
                    for step in range(self.predict_steps):
                        window_scaled = scaler.fit_transform(np.array(window).reshape(-1, 1))
                        with torch.no_grad():
                            output = self.trend_predictor(torch.Tensor(window_scaled).unsqueeze(0).squeeze(-1))
                        window.append(scaler.inverse_transform(output).item())
                        window = window[1:]
                    break
            if found:
                return [window_scaled[-1-self.predict_steps][0], output.item()]
        curr_price = df.iloc[0]
        return [curr_price, curr_price]
            
        
    def _next_observation(self):
        
        norm_balance = self.balance / self.initial_balance
        norm_price, norm_pred = self.get_observation()
        norm_shares = self.shares_held * self.initial_price / self.initial_balance
        
        obs = np.array([norm_balance, norm_price, norm_pred, norm_shares]).astype(np.float32)
        
        return obs

    def step(self, action):
        # Parse action
        action_type = int(np.floor(action[0]))
        action_type = np.clip(action_type, 0, 2)  # 0: Buy, 1: Hold, 2: Sell
        action_param = np.clip(action[1], 0.0, 1.0)

        current_price = self.df.iloc[self.current_step]
        prev_value = self.balance + self.shares_held * current_price

        if action_type == 0:  # BUY
            amount_to_spend = self.balance * action_param
            if amount_to_spend > 0 and current_price > 0:
                shares_bought = amount_to_spend / current_price
                self.balance -= amount_to_spend
                self.shares_held += shares_bought
        elif action_type == 2:  # SELL
            shares_to_sell = self.shares_held * action_param
            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell

        self.current_step += 1

        if self.current_step >= len(self.df):
            new_price = current_price
            done = True
        else:
            new_price = self.df.iloc[self.current_step]
            done = self.current_step >= len(self.df) - 1
            
        current_value = self.balance + self.shares_held * new_price
        
        reward = (current_value - prev_value) - (self.balance * self.penalty_coef)

        return self._next_observation(), reward, done, {}

    def render(self, mode='human'):
        current_price = self.df.iloc[self.current_step]
        value = self.balance + self.shares_held * current_price
        print(f'Balance: {self.balance:.2f}')
        print(f'Shares: {self.shares_held:.2f}')
        print(f'Portfolio Value: {value:.2f}')