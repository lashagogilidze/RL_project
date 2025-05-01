import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import pickle

import Actor

# environment-ს ვალოადებ და render = human mode რათა სიმულაცია დავინახო
env = gym.make('Ant-v4', render_mode="human").unwrapped


# ნასწავლ აქტორს ვტვირთავ
with open('actor_object.pkl', 'rb') as f:
    actor = pickle.load(f)

reward_single=0
observation,info = env.reset() 

# ვუშვებ 2000 timestep-ს
for timestep in range(2000):
        
    # მომაქ action actor-დან და ვააბდეითებ environment-ს
    action=actor.select_action(observation).detach().numpy()
        
    observation_next, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

    observation = observation_next
    reward_single += reward

#გამომაქ reward
print(reward_single)
env.close()