from custom_env import StockTradingEnv
import pandas as pd
from stable_baselines3 import PPO

df = pd.read_csv('dataset/stocks/A.csv')

env = StockTradingEnv(df)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        break
env.render()