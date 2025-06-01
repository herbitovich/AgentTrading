from custom_env import StockTradingEnv
import pandas as pd
import torch
import torch.nn as nn

class TrendPredictor(nn.Module):
    def __init__(self, window_size, hidden_size):
        super(TrendPredictor, self).__init__()
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(self.window_size, self.hidden_size, batch_first=True)
        self.do = nn.Dropout(0.1)
        self.fc = nn.Linear(self.hidden_size, 1)
    def forward(self, x):
        x, _ = self.LSTM(x)
        x = self.do(x)
        x = self.fc(x)
        return x

WINDOW_SIZE = 256
HIDDEN_SIZE = 512

trend_predictor = TrendPredictor(WINDOW_SIZE, HIDDEN_SIZE)
trend_predictor.load_state_dict(torch.load("LSTM_Trader.pt"))
from stable_baselines3 import SAC
from sklearn.model_selection import train_test_split
import os

train, test = train_test_split(os.listdir('dataset/stocks'), test_size=0.01)

model = SAC("MlpPolicy", env=StockTradingEnv(pd.read_csv(os.path.join('dataset/stocks', train[0]))["Close"], trend_predictor, 5), verbose=1, device='cuda')
for ds in train:
    print(f"Started training on {ds}") 
    ds_path = os.path.join('dataset/stocks', ds)
    df = pd.read_csv(ds_path)["Close"]
    if len(df)<=WINDOW_SIZE:
        print("Dataset is too short.")
        continue
    env = StockTradingEnv(df, trend_predictor, 5)
    model.set_env(env)
    model.learn(total_timesteps=2048)
    print("Finished training")
for test_ds in test:
    print(f"Evaluating on {test_ds}")
    df = pd.read_csv(os.path.join('dataset/stocks', test_ds))["Close"]
    if len(df)<=WINDOW_SIZE:
        print("Dataset is too short.")
        continue
    env = StockTradingEnv(df, trend_predictor, 5)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            break
    env.render()
model.save("agent-trader")