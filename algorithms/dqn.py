import gymnasium as gym
import torch.nn as nn
import torch
import random
import numpy as np
from collections import deque
from time import sleep, time


class QNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_nn = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2) 
        )
        
    def forward(self, s):
        x = self.s_nn(s)
        return x

def preprocess_state(s):
    cart_position = s[0]#s[0] / 4.8
    cart_velocity = s[1]#s[1]/ 4
    pole_angle = s[2]#s[2] / 2.4
    pole_velocity = s[3]#s[3] / 0.2095
    return torch.tensor([cart_position, cart_velocity, pole_angle, pole_velocity], dtype=torch.float32)


def train_batch(q_net, target_net, batch, optimizer, criterion, gamma):
    ys = torch.tensor([])
    qs = torch.tensor([])
            
    optimizer.zero_grad()
    for k in range(len(batch)):
        s_t, a, r, s_t_1, terminated = batch[k]
        with torch.no_grad():
            ys = torch.cat((ys, torch.tensor([r + int(not(terminated))*gamma*torch.max(target_net(s_t_1))])))
        qs = torch.cat((qs, q_net(s_t)[a].reshape(1)))
                    
    loss = criterion(qs, ys)
    loss.backward()
    optimizer.step()

def dqn(n_episodes=600, gamma=0.99, eps=0.5, eps_decay=0.99, buffer_size=3000, batch_size=100):
    env = gym.make("CartPole-v1", render_mode="rgb_array").unwrapped
    q_net = QNN()
    target_net = QNN()
    target_net.load_state_dict(q_net.state_dict())
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(q_net.parameters())
    buffer = deque(maxlen=buffer_size)
    results = []
    times = []
    
    last_time = time()
    start = time()
    for i in range(n_episodes): 
        print(f"dqn episode {i}")
        env.reset(seed=123, options={"low": -0.1, "high": 0.1})
        #print(len(buffer))
        for _ in range(2000):
            s_t = preprocess_state(env.state)
            if random.random() < eps:
                a = random.choice([0, 1])
            else:
                a = torch.argmax(q_net(s_t)).item()
                
            s_t_1, r, terminated, _, _ = env.step(a)
            s_t_1 = preprocess_state(s_t_1)
            
            buffer.append((s_t, a, r, s_t_1, terminated))
            
            batch = random.sample(buffer, min(len(buffer), batch_size))
            
            train_batch(q_net, target_net, batch, optimizer, criterion, gamma)
            
            if terminated:
                env.reset(seed=123, options={"low": -0.1, "high": 0.1})
        
        eps = eps*eps_decay    
        if i % 2:  
            target_net.load_state_dict(q_net.state_dict())
        torch.save(q_net.state_dict(), "models/model_dqn.pth")
        if time() - last_time > 20:
            last_time = time()
            results.append(play())
            times.append(time()-start)
            print(results[-1])
            if results[-1] >= 1999:
                break
            
    torch.save(q_net.state_dict(), "models/model_dqn.pth")
    return times, results
    
def play():
    env = gym.make("CartPole-v1", render_mode="rgb_array").unwrapped
    r_sum = 0
    for _ in range(10):
        s, _ = env.reset(seed=123, options={"low": -0.1, "high": 0.1})
        net = QNN()
        net.load_state_dict(torch.load("models/model_dqn.pth", weights_only=True))
        terminated = False
        for _ in range(2000):
            a_values = net(preprocess_state(s))
            a = torch.argmax(a_values)
            s_t_1, r, terminated, _, _ = env.step(a.item())
            r_sum += r
            s = s_t_1
            if terminated:
                break
    return r_sum/10