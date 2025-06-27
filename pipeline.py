import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import gymnasium
from stable_baselines3.common.monitor import Monitor
import requests
import random
import register_env

class TrendPredictor(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(TrendPredictor, self).__init__()
        self.fc = nn.Linear(hidden_size, 1)
        self.do = nn.Dropout(0.1)
        self.LSTM = nn.LSTM(num_features, hidden_size, batch_first=True)

    def forward(self, x):
        out, _ = self.LSTM(x)  # out: (batch, seq_len, hidden_size)
        out = self.do(out)
        return nn.functional.sigmoid(self.fc(out[:, -1, :]))  # только последний таймстеп интересует

WINDOW_SIZE = 30
HIDDEN_SIZE = 64

trend_predictor = TrendPredictor(num_features=12, hidden_size=HIDDEN_SIZE)
trend_predictor.load_state_dict(torch.load("new_predictor_best.pt"))

from stable_baselines3 import SAC   
import os
import threading
import time

test = [os.path.join('trader-agent', 'test', file) for file in os.listdir('trader-agent/test')]

model = SAC.load("best_model", device='cpu')

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

def trade(df, agent_name, company, trend_predictor, model=None):
    """Trade function modified to send transactions"""
    try:
        test_env = gymnasium.make("gymnasium_env/Trading-v1", dfs=df, trend_predictor=trend_predictor, render_mode='human')
    except RuntimeError:
        print("Dataset is too short.")
        return None
    
    obs, info = test_env.reset()
    done = False

    while not done:
        current_price = info["current_price"]
        date = info["current_date"]
        
        if model:
            action, _states = model.predict(obs, deterministic=True)
        else:
            action = np.random.randint(1, 100, (3))/100

        action_idx = np.argmax(action)
        action_amount = action[action_idx]
        action_type = {0: "buy", 1: "hold", 2: "sell"}[action_idx]

        agent_data = [{
            "agent": agent_name,
            "action": action_type,
            "amount": float(action_amount),
            "value": info["value"]
        }]
        
        # Send transaction data
        send_transaction(
            company=company,
            current_price=float(current_price),
            date=str(date),
            agent_actions=agent_data
        )
        
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        
        time.sleep(5)
    
def start_thread(file_path, model_trained, trend_predictor):
    """Start trading thread for a stock file"""
    company = os.path.splitext(os.path.basename(file_path))[0]
    
    trained_thread = threading.Thread(
        target=trade,
        args=([file_path], "Trained trader", company, trend_predictor, model_trained)
    )   
    
    untrained_thread = threading.Thread(
        target=trade,
        args=([file_path], "Untrained trader", company, trend_predictor)
    )
    
    trained_thread.start()
    untrained_thread.start()
    
    return trained_thread, untrained_thread

def main():
    threads = []
    for file in test[:20]:
        t1, t2 = start_thread(file, model, trend_predictor)
        threads.extend([t1, t2])

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()