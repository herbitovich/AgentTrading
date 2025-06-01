import pandas as pd
from custom_env import StockTradingEnv
import torch
import torch.nn as nn
import numpy as np
import requests  # For sending POST requests
import json

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
import os
import random
import threading
import time

# Load models
model_trained = SAC.load("agent-trader", device='cpu')
model_untrained = SAC("MlpPolicy", env=StockTradingEnv(pd.read_csv(os.path.join('dataset/stocks', random.choice(os.listdir('dataset/stocks'))))["Close"], trend_predictor, 5), device='cpu')

def send_transaction(company, current_price, date, agent_actions):
    """Send transaction data to localhost:8000"""
    url = "http://localhost:8000/api/post-trades/"
    payload = {
        "company": company,
        "current_price": current_price,
        "date": date,
        "agents": agent_actions
    }
    try:
        response = requests.post(url, json=payload, timeout=5)
        print(f"Status: {response.status_code}, Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send request: {str(e)}")

def trade(df, model, model_name, company, trend_predictor):
    """Trade function modified to send transactions"""
    if len(df) <= WINDOW_SIZE:
        print("Dataset is too short.")
        return None
        
    env = StockTradingEnv(df, trend_predictor, 5)
    obs = env.reset()
    done = False
    step_count = WINDOW_SIZE
    
    while not done:
        current_price = df.iloc[env.current_step]
        date = df.index[env.current_step]
        
        action, _states = model.predict(obs, deterministic=True)
        
        action_idx = np.argmax(action)
        action_amount = action[action_idx]
        action_type = {0: "buy", 1: "hold", 2: "sell"}[action_idx]
        
        agent_data = [{
            "agent": model_name,
            "action": action_type,
            "amount": float(action_amount),
            "value": env.balance+env.shares_held*current_price
        }]
        
        # Send transaction data
        send_transaction(
            company=company,
            current_price=float(current_price),
            date=str(date),
            agent_actions=agent_data
        )
        
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        time.sleep(5)
    
    # env.render()  # Keep if needed for visualization

def process_data(file_path):
    """Process data including datetime index"""
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return df["Close"]

def start_thread(file_path, model_trained, model_untrained, trend_predictor):
    """Start trading thread for a stock file"""
    df = process_data(file_path)
    company = os.path.splitext(os.path.basename(file_path))[0]
    
    trained_thread = threading.Thread(
        target=trade,
        args=(df, model_trained, "Trained trader", company, trend_predictor)
    )
    
    untrained_thread = threading.Thread(
        target=trade,
        args=(df, model_untrained, "Untrained trader", company, trend_predictor)
    )
    
    trained_thread.start()
    untrained_thread.start()
    
    return trained_thread, untrained_thread

def main():
    threads = []
    for file_name in os.listdir('dataset/stocks')[:3]:
        if file_name.endswith('.csv'):
            file_path = os.path.join('dataset/stocks', file_name)
            t1, t2 = start_thread(file_path, model_trained, model_untrained, trend_predictor)
            threads.extend([t1, t2])
    
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()