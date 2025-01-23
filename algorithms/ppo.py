import gymnasium as gym
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import StepLR
from torch.distributions import Categorical
import random
import numpy as np
from time import sleep, time
from datetime import datetime

class NNActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.simple_nn = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.classifier_p = nn.Sequential(
            nn.Linear(4, 2),
            nn.Softmax(dim=-1))
        
    def forward(self, s):
        x_1 = self.simple_nn(s)
        return self.classifier_p(x_1)
    
class NNCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.simple_nn = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.classifier_v = nn.Sequential(
            nn.Linear(4, 1))
        
    def forward(self, s):
        x_1 = self.simple_nn(s)
        return self.classifier_v(x_1)
    
def actor_loss(p, adv, entropy, eps):
    loss = -torch.min(p*adv, torch.clamp(p, 1-eps, 1+eps)*adv)
    # Współczynnik 0.1 dobrany doświadczalnie
    entropy_penalty = -0.1*entropy
    #print(f"Loss: {loss.item()}")
    #print(f"entropy_penalty {entropy_penalty.item()}")
    return loss + entropy_penalty

def critic_loss(y, v):
    loss = (y-v)**2
    return loss

def preprocess_state(s):
    cart_position = s[0]#s[0] / 4.8
    cart_velocity = s[1]#s[1]/ 4
    pole_angle = s[2]#s[2] / 2.4
    pole_velocity = s[3]#s[3] / 0.2095
    return torch.tensor([cart_position, cart_velocity, pole_angle, pole_velocity], dtype=torch.float32)

def ppo(n_episodes=3000, gamma=0.99, Ne=3, eps=0.2):
    env = gym.make("CartPole-v1", render_mode="rgb_array").unwrapped
    actor = NNActor()
    critic = NNCritic()
    # nie zwiększamy eps
    a_optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
    c_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)
    results = []
    times = []
    
    start = time()
    last_time = time()
    
    for i in range(n_episodes):
        print(f"ppo episode {i}")
        env.reset(seed=123, options={"low": -0.1, "high": 0.1})
        for _ in range(2000):
            s_t = preprocess_state(env.state)
            
            distribution = actor(s_t)
            distribution_cat = Categorical(distribution)
            v_s_t = critic(s_t)
                
            a = distribution_cat.sample()
            s_t_1, r, terminated, _, _ = env.step(a.item())
            
            actor_beta = NNActor()
            
            actor_beta.load_state_dict(actor.state_dict())
            for e in range(Ne):
                
                if terminated:
                    adv = r - v_s_t
                    y = r
                else:
                    with torch.no_grad():
                        v_s_t_1 = critic(preprocess_state(s_t_1))
                    adv = r + gamma*v_s_t_1 - v_s_t
                    y = r + gamma*v_s_t_1
                
                a_optimizer.zero_grad()
                dist_b = actor_beta(s_t)
                p = distribution[a] / dist_b[a].detach()
                a_loss = actor_loss(p, adv.detach(), distribution_cat.entropy(), eps)
                a_loss.backward()
                a_optimizer.step()
                    
                c_optimizer.zero_grad()
                c_loss = critic_loss(y, v_s_t)
                c_loss.backward()
                c_optimizer.step()
                
                distribution = actor(s_t)
                distribution_cat = Categorical(distribution)
                v_s_t = critic(s_t)
            
            if terminated:
                env.reset(seed=123, options={"low": -0.1, "high": 0.1})
            
        torch.save(actor.state_dict(), "models/model_ppo.pth")
        if time() - last_time > 20:
            results.append(play())
            last_time = time()
            times.append(time()-start)
            if results[-1] > 1998:
                break
            #print(results[-1])
        
     
    torch.save(actor.state_dict(), "models/model_ppo.pth")
    return times, results
    
def play():
    env = gym.make("CartPole-v1", render_mode="rgb_array").unwrapped
    r_sum = 0
    for _ in range(10):
        env.reset(seed=123, options={"low": -0.1, "high": 0.1})
        net = NNActor()
        net.load_state_dict(torch.load("models/model_ppo.pth", weights_only=True))
        terminated = False
        for _ in range(2000):
            with torch.no_grad():
                distribution = net(preprocess_state(env.state))
            a = torch.multinomial(distribution, 1)
            s_t_1, r, terminated, _, _ = env.step(a.item())
            r_sum += r
            if terminated:
                break
    return r_sum/10