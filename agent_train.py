import register_env
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium
from stable_baselines3.common.monitor import Monitor

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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os

checkpoint = CheckpointCallback(save_freq=50000, save_path='./logs/')

train = [os.path.join('trader-agent', 'train', file) for file in os.listdir('trader-agent/train')]
test, eval = train_test_split([os.path.join('trader-agent', 'test', file) for file in os.listdir('trader-agent/test')], test_size=0.2, random_state=97)


train_env = gymnasium.make("gymnasium_env/Trading-v1", dfs=train, trend_predictor=trend_predictor, render_mode='human')
eval_env = Monitor(gymnasium.make("gymnasium_env/Trading-v1", dfs=eval, trend_predictor=trend_predictor, render_mode='human'))
test_env = Monitor(gymnasium.make("gymnasium_env/Trading-v1", dfs=test, trend_predictor=trend_predictor, render_mode='human'))


eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=5000,
                             deterministic=False, render=False)


model = SAC("MlpPolicy", env=train_env, verbose=1, device='cpu')
model.learn(total_timesteps=1_000_000, callback=[checkpoint, eval_callback], progress_bar=True)
model.save("agent-trader")
mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=len(test), deterministic=True, render=False)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")