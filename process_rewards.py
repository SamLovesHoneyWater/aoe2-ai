import matplotlib.pyplot as plt
import numpy as np

K = 0.75

def smooth_rewards(rewards, window_size=10):
    return np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

def discretize_rewards(rewards):
    discretized_rewards = np.zeros(len(rewards))
    for i in range(1, len(rewards)):
        if rewards[i] != rewards[i - 1]:
            discretized_rewards[i] = rewards[i] - rewards[i - 1]
    discretized_rewards[-1] = -100
    return discretized_rewards

def decaying_rewards(rewards, decay_rate=K):
    decayed_rewards = np.zeros(len(rewards))
    decayed_rewards[0] = rewards[0]
    for i in range(1, len(rewards)):
        decayed_rewards[i] = rewards[i] + decay_rate * decayed_rewards[i - 1]
    return decayed_rewards

def forward_looking_rewards(rewards, decay_rate=K):
    decayed_rewards = np.zeros(len(rewards))
    decayed_rewards[-1] = rewards[-1]
    for i in range(len(rewards)-2, -1, -1):
        decayed_rewards[i] = rewards[i] + decay_rate * decayed_rewards[i + 1]
    return decayed_rewards

def idle_punishment(rewards, idle_penalty=-1, idle_threshold=25):
    idle = np.zeros(len(rewards))
    consecutive_idles = 0
    for i in range(0, len(rewards)):
        if rewards[i] == 0:
            consecutive_idles += 1
            idle[i] = min(0, idle_penalty * (consecutive_idles - idle_threshold))
        else:
            consecutive_idles = 0
    return idle

def process_rewards(scores):
    scores = discretize_rewards(scores)
    smooth1 = forward_looking_rewards(scores)
    smooth2 = decaying_rewards(scores)
    idle = idle_punishment(scores)
    smooth = smooth1 + smooth2 - scores + idle
    return smooth