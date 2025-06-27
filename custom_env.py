import gymnasium
import numpy as np
from gymnasium import spaces
import torch
import pandas as pd

class StockTradingEnv(gymnasium.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, dfs, trend_predictor, initial_balance=10000, penalty_coef=0.5, window_size=30, bankrupt_coef=0.5, commission=0.001, render_mode=None):
        super(StockTradingEnv, self).__init__()
        self.trend_predictor = trend_predictor
        self.trend_predictor.eval()
        self.dfs = np.array(dfs) # dataframes, not depth-first search
        if self.dfs.ndim > 1:
            return AttributeError("The dataframes URLs array must be one-dimensional.")

        self.initial_balance = initial_balance
        self.bankrupt_coef = bankrupt_coef
        self.penalty_coef = penalty_coef
        self.commission = commission
        self.window_size = window_size

        self._precompute()

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float64
        )

        self.observation_space = spaces.Box(
            low=np.concat((np.array([0]*29), np.array([-np.inf]*4))), 
            high=np.concat((np.array([np.inf]*7), np.array([1]*22), np.array([np.inf]*4))), 
            shape=(33,), 
            dtype=np.float64
        )
        self.reset()

    def _get_info(self):
        current_price = self.prices[self.index][self.current_step]
        return {
            "value": self.balance + self.shares_held * current_price,
            "current_step": self.current_step,
            "current_price": current_price,
            "current_date": self.dates[self.index][self.current_step]
        }
    
    def _precompute(self):
        df_index = 0
        self.data = []
        self.prices = []
        self.predictions = []
        self.dates = []
        while df_index < len(self.dfs):
            print(f"Precomputed {df_index}/{len(self.dfs)}")
            df = pd.read_csv(self.dfs[df_index])
            if len(df) > self.window_size + 1:
                dates = df['date']
                df = df.drop(columns=['date'], errors='ignore')
                prices = df["price"].values
                df = df.drop(columns=["price"]).values
                self.data.append(df)
                self.prices.append(prices)
                self.dates.append(dates)

                windows = []
                step = self.window_size
                while step < len(df):
                    window = df[step - self.window_size : step]
                    windows.append(window)
                    step+=1
                windows = np.array(windows, dtype=np.float32)
                windows = torch.from_numpy(windows)
                with torch.no_grad():
                    preds = self.trend_predictor(windows)
                preds = torch.round(preds).view(-1)
                self.predictions.append(preds.cpu().numpy())
            df_index += 1
        self.indicies = list(np.arange(len(self.data)))
        if not self.data:
            raise RuntimeError("No suitable dataset found.")

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_mult = 1
        self.index = np.random.choice(self.indicies)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _get_obs(self):
        norm_balance = self.balance / self.initial_balance
        weekday, month, quarter = np.array(self.data[self.index][self.current_step][2:5], dtype=np.int8)
        weekday = np.eye(5)[weekday]
        month = np.eye(12)[month-1]
        quarter = np.eye(4)[quarter-1]
        current_price = self.prices[self.index][self.current_step]
        shares_value = (self.shares_held * current_price) / self.initial_balance

        pred_index = self.current_step - self.window_size
        grow_pred = self.predictions[self.index][pred_index]
        return np.concat((np.array([norm_balance, shares_value], dtype=np.float64), np.array(self.data[self.index][self.current_step][:2]), np.array(self.data[self.index][self.current_step][-3:]), weekday, month, quarter, np.array([grow_pred]), np.array(self.data[self.index][self.current_step][5:-3])), axis=0)

    def step(self, action):
        """
        range награды = [-21, 30]
        min = -10-1-10 = -21 (ценность портфеля упала до 0,
        idle fraction = 0,(99) = 1, -10 за банкротство)

        max = 20-0+10 = 20 (ценность выросла вдвое и больше,
        весь баланс в акциях)
        """
        current_price = self.prices[self.index][self.current_step]
        prev_value = self.balance + self.shares_held * current_price

        action_type = np.argmax(action)
        action_amount = action[action_type]

        if action_type == 0:  # BUY
            amount_to_spend = self.balance * action_amount
            shares_bought = amount_to_spend / current_price
            self.balance -= amount_to_spend + amount_to_spend * self.commission
            self.shares_held += shares_bought
        elif action_type == 2:  # SELL
            shares_to_sell = self.shares_held * action_amount
            sale_amount = shares_to_sell * current_price
            sale_amount *= (1 - self.commission)
            self.balance += sale_amount
            self.shares_held -= shares_to_sell

        self.current_step += 1
        
        truncated = self.current_step >= len(self.data[self.index]) - 1

        new_price = self.prices[self.index][self.current_step]
        current_value = self.balance + self.shares_held * new_price
        
        # это pct_change/10, для хорошего масштаба наград
        reward = ((current_value - prev_value) / prev_value) * 10 
        # подразумеваем |pct_change| <= 200, больше уже удачные обстоятельства
        reward = np.clip(reward, -10, 20) 

        # наказание за неинвестированный капитал
        # (на некоторых стоках действительно лучше холдить)
        if action_type != 1:
            idle_fraction = self.balance / current_value
            reward -= idle_fraction * self.penalty_coef

        terminated = False
        if current_value <= self.bankrupt_coef * self.initial_balance: # обанкротился
            terminated = True
            # -10, т.к. удваиваем наказание pct_change (потеряли 100%)
            reward -= 10

        # награда за умножение значения портфеля
        if current_value//self.initial_balance > self.current_mult:
            reward += 10
            self.current_mult += 1

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        current_price = self.prices[self.index][self.current_step]
        value = self.balance + self.shares_held * current_price
        print(f'Step: {self.current_step}/{len(self.prices)} | Balance: ${self.balance:.2f} | '     
              f'Shares: {self.shares_held:.2f} | Value: ${value:.2f}')