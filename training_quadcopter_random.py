import gym
import torch

from stable_baselines3 import PPO

from quadcopter_model_random import Quadcopter_Random

print(torch.version.cuda)
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

env = Quadcopter_Random()
env.reset()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2000000)

# model.save("ppo_quadcopter_rollpitch_random")
print("Done")