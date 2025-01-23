import gymnasium as gym
import torch.nn as nn
from torch.distributions import Categorical
import torch
import random
import numpy as np
from time import sleep, time

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

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
        
        self.simple_nn.apply(init_weights)
        self.classifier_p.apply(init_weights)
        
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
        
        self.simple_nn.apply(init_weights)
        self.classifier_v.apply(init_weights)
        
    def forward(self, s):
        x_1 = self.simple_nn(s)
        return self.classifier_v(x_1)
    
def actor_loss(adv, log_prob, entropy):
    # Współcynnik 0.1 dobrany doświadczalnie
    loss = -adv*log_prob - 0.1*entropy
    return loss

def critic_loss(y, v):
    loss = torch.float_power(y - v, 2)
    return loss

def preprocess_state(s):
    cart_position = s[0]#s[0] / 4.8
    cart_velocity = s[1]#s[1]/ 4
    pole_angle = s[2]#s[2] / 2.4
    pole_velocity = s[3]#s[3] / 0.2095
    return torch.tensor([cart_position, cart_velocity, pole_angle, pole_velocity], dtype=torch.float32)

def a2c(n_episodes=4000, gamma=0.99):
    env = gym.make("CartPole-v1", render_mode="rgb_array").unwrapped
    actor = NNActor()
    critic = NNCritic()
    # Zwiększamy eps - skutkuje lepszą stabilnością numeryczną
    a_optimizer = torch.optim.Adam(actor.parameters(), lr=0.001, eps=1e-5)
    c_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001, eps=1e-5)
    
    results = []
    times = []
    
    last_time = time()
    start = time()
    
    for j in range(n_episodes):
        #if j % 10 == 0:
        #    target_critic.load_state_dict(critic.state_dict())
        print(f"a2c episode {j}")
        env.reset(seed=123, options={"low": -0.1, "high": 0.1})
        for i in range(2000):
            
            s_t = preprocess_state(env.state)
            
            distribution = actor(s_t)
            distribution = Categorical(distribution)
            v_s_t = critic(s_t)
            a = distribution.sample()
            s_t_1, r, terminated, _, _ = env.step(a.item())
                
            if terminated:
                adv = r - v_s_t 
                y = r
            else:
                with torch.no_grad():
                    v_s_t_1 = critic(preprocess_state(s_t_1))
                adv = r + gamma*v_s_t_1 - v_s_t
                y = r + gamma*v_s_t_1
            
            
            a_optimizer.zero_grad()
            a_loss = actor_loss(adv.detach(), distribution.log_prob(a), distribution.entropy())
            a_loss.backward()
            #torch.nn.utils.clip_grad_norm_(actor.parameters(), 2000)
            a_optimizer.step()
                
            c_optimizer.zero_grad()
            c_loss = critic_loss(y, v_s_t)
            c_loss.backward()
            #torch.nn.utils.clip_grad_norm_(critic.parameters(), 2000)
            #c_grad.append(critic.classifier_v[0].weight.grad.norm().item())
            c_optimizer.step()
            
            if terminated:
                env.reset(seed=123, options={"low": -0.1, "high": 0.1})
        
        torch.save(actor.state_dict(), "models/model_a2c.pth") 
        if time() - last_time > 20:
            last_time = time()
            results.append(play())
            times.append(time()-start)
            if results[-1] > 1999:
                break
     
    
    return times, results
    
    
def play():
    env = gym.make("CartPole-v1", render_mode="rgb_array").unwrapped
    r_sum = 0
    for _ in range(10):
        env.reset(seed=123, options={"low": -0.1, "high": 0.1})
        net = NNActor()
        net.load_state_dict(torch.load("models/model_a2c.pth", weights_only=True))
        terminated = False
        for _ in range(2000):
            distribution = net(preprocess_state(env.state))
            distribution = Categorical(distribution)
            a = distribution.sample()
            s_t_1, r, terminated, _, _ = env.step(a.item())
            r_sum += r
            if terminated:
                break
    return r_sum/10