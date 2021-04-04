import torch
import numpy as np
import matplotlib.pyplot as plt

def normalize(data):
    data_normalized = (data - data.mean()) / (data.std() + 1e-6)
    return data_normalized   

def prepro_half(I):
    I = I[35:195] # crop
    I = I[::2,::2, 0]
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I

def prepro_crop(I):
    I = I[35:195] 
    return I

def prepo_full(I):
    I = I[35:195] # crop
    I = I[:,:, 0]
    I[I == 144] = 1 # erase background (background type 1)
    I[I == 109] = 1 # erase background (background type 2)
    I[I != 0] = 0 # everything else (paddles, ball) just set to 1
    return I

def prepo_full_one_dim(I):
    I = prepo_full(I)
    I = I.astype(np.float32).ravel()
    I = I / 255.0
    return I

def prepro_half_one_dim(I):
    I = prepro_half(I)
    I = I.astype(np.float32).ravel()
    return I

def prepo_crop(I):
    I = I[35:195] # crop
    I = I / 255.0
    return I

def plot(datas):
    print('----------')

    plt.plot(datas)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Datas')
    plt.show()

    print('Max :', np.max(datas))
    print('Min :', np.min(datas))
    print('Avg :', np.mean(datas))

def new_std_from_rewards(rewards, reward_target):
    rewards     = np.array(rewards)
    mean_reward = np.mean(reward_target - rewards)
    new_std     = mean_reward / reward_target

    if new_std < 0.25:
        new_std = 0.25
    elif new_std > 1.0:
        new_std = 1.0

    return new_std