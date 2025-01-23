import gymnasium as gym
import argparse
import torch
import torch.nn as nn
from algorithms.dqn import QNN, preprocess_state
from algorithms.ppo import NNActor, NNCritic

parser = argparse.ArgumentParser()

parser.add_argument('-o', "--opts",) 

def see_dqn():
    env = gym.make("CartPole-v1", render_mode="human").unwrapped
    env.reset(seed=123, options={"low": -0.1, "high": 0.1})
    net = QNN()
    net.load_state_dict(torch.load("models/model_dqn.pth", weights_only=True))
    terminated = False
    r_sum = 0
    while not terminated:
        a = torch.argmax(net(preprocess_state(env.state))).item()
        s_t_1, r, terminated, _, _ = env.step(a)
        r_sum += r
    print(r_sum)
    
def see_a2c():
    env = gym.make("CartPole-v1", render_mode="human").unwrapped
    env.reset(seed=123, options={"low": -0.1, "high": 0.1})
    net = NNActor()
    net.load_state_dict(torch.load("models/model_a2c.pth", weights_only=True))
    terminated = False
    r_sum = 0
    while not terminated:
        distribution = net(preprocess_state(env.state))
        a = torch.multinomial(distribution, 1)
        s_t_1, r, terminated, _, _ = env.step(a.item())
        r_sum += r
    print(r_sum)
    
def see_ppo():
    env = gym.make("CartPole-v1", render_mode="human").unwrapped
    env.reset(seed=123, options={"low": -0.1, "high": 0.1})
    net = NNActor()
    net.load_state_dict(torch.load("models/model_ppo.pth", weights_only=True))
    terminated = False
    r_sum = 0
    while not terminated:
        distribution = net(preprocess_state(env.state))
        a = torch.multinomial(distribution, 1)
        s_t_1, r, terminated, _, _ = env.step(a.item())
        r_sum += r
    print(r_sum)
    
if __name__ == "__main__":
    args = parser.parse_args()
    opts = args.opts
    if opts == "a2c":
        see_a2c()
    elif opts == "dqn":
        see_dqn()
    elif opts == "ppo":
        see_ppo()
    else:
        print("Invalid option")