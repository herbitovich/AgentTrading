import gym
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from gym import spaces
import torch

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, trend_predictor, predict_steps=5, initial_balance=10000, penalty_coef=0.01):
        super(StockTradingEnv, self).__init__()
        self.trend_predictor = trend_predictor
        if self.trend_predictor: 
            self.trend_predictor.eval()
        self.predict_steps = predict_steps
        self.df = df.reset_index(drop=True).values
        self.initial_balance = initial_balance
        self.penalty_coef = penalty_coef
        self.window_size = trend_predictor.window_size

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(3,), 
            dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.prev_value = self.initial_balance
        return self._next_observation()

    def get_observation(self):
        window = self.df[self.current_step - self.window_size : self.current_step]
        scaler = MinMaxScaler()
        window_scaled = scaler.fit_transform(window.reshape(-1, 1))
        
        with torch.no_grad():
            for step in range(self.predict_steps):
                output = self.trend_predictor(torch.Tensor(window_scaled).unsqueeze(0).squeeze(-1)).item()
                
                window_scaled = np.roll(window_scaled, -1, axis=0)
                window_scaled[-1] = np.array([output])
        return [window_scaled[-self.predict_steps-1][0], output]
            
    def _next_observation(self):
        norm_balance = self.balance / self.initial_balance
        norm_price, norm_pred = self.get_observation()
        return np.array([norm_balance, norm_price, norm_pred], dtype=np.float32)

    def step(self, action):
        current_price = self.df[self.current_step]
        prev_value = self.balance + self.shares_held * current_price

        action_type = np.argmax(action)
        action_amount = action[action_type]

        if action_type == 0:  # BUY
            amount_to_spend = self.balance * action_amount
            shares_bought = amount_to_spend / current_price
            self.balance -= amount_to_spend
            self.shares_held += shares_bought
        elif action_type == 2:  # SELL
            shares_to_sell = self.shares_held * action_amount
            sale_amount = shares_to_sell * current_price
            self.balance += sale_amount
            self.shares_held -= shares_to_sell

        self.current_step += 1
        
        done = self.current_step >= len(self.df) - 1
        
        new_price = self.df[self.current_step]
        current_value = self.balance + self.shares_held * new_price
        
        reward = (current_value - prev_value) / (prev_value + 1e-8)
        reward -= abs(action_amount) * self.penalty_coef

        return self._next_observation(), reward, done, {}

    def render(self, mode='human'):
        current_price = self.df[self.current_step]
        value = self.balance + self.shares_held * current_price
        print(f'Step: {self.current_step}/{len(self.df)} | Balance: ${self.balance:.2f} | '     
              f'Shares: {self.shares_held:.2f} | Value: ${value:.2f}')