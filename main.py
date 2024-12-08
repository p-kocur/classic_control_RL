import gymnasium as gym
import torch.nn as nn
import torch
import random
import numpy as np
from collections import deque

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NNActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.simple_nn = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.classifier_p = nn.Sequential(
            nn.Linear(8, 2),
            nn.Softmax(dim=-1))
        
    def forward(self, s):
        x_1 = self.simple_nn(s)
        return self.classifier_p(x_1)
    
class NNCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.simple_nn = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.classifier_v = nn.Sequential(
            nn.Linear(8, 1))
        
    def forward(self, s):
        x_1 = self.simple_nn(s)
        return self.classifier_v(x_1)
    
def actor_loss(adv, pi):
    loss = -adv*torch.log(pi)
    return loss

def critic_loss(y, v):
    loss = (y-v)**2
    return loss

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

def max_move_value(net, s):
    with torch.no_grad():
        moves = [net(s, torch.tensor([a], dtype=torch.float32)) for a in [0, 1]]
    best_move = np.max(moves)
    return best_move

def max_move(net, s):
    with torch.no_grad():
        moves = [net(s, torch.tensor([a], dtype=torch.float32)) for a in [0, 1]]
    best_move = np.argmax(moves)
    return best_move


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

def dqn(n_episodes=1000, gamma=0.99, eps=0.5, eps_decay=0.95, buffer_size=3000, batch_size=100):
    env = gym.make("CartPole-v1", render_mode="rgb_array").unwrapped
    q_net = QNN()
    target_net = QNN()
    target_net.load_state_dict(q_net.state_dict())
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(q_net.parameters())
    buffer = deque(maxlen=buffer_size)
    
    for i in range(n_episodes): 
        print(i)
        env.reset(seed=123, options={"low": -0.1, "high": 0.1})
        #print(len(buffer))
        for _ in range(1000):
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
                break
        
        eps = eps*eps_decay    
        if i % 2:  
            target_net.load_state_dict(q_net.state_dict())
            
    torch.save(q_net.state_dict(), "control/models/model_dqn.pth")
            
            

def a2c(n_episodes=4000, gamma=0.99):
    env = gym.make("CartPole-v1", render_mode="rgb_array").unwrapped
    actor = NNActor()
    critic = NNCritic()
    a_optimizer = torch.optim.Adam(actor.parameters(), weight_decay=0.01)
    c_optimizer = torch.optim.Adam(critic.parameters(), weight_decay=0.01)
    
    for _ in range(n_episodes):
        env.reset(seed=123, options={"low": -0.1, "high": 0.1})
        for i in range(1000):
            
            s_t = preprocess_state(env.state)
            
            distribution = actor(s_t)
            v_s_t = critic(s_t)
                
            a = torch.multinomial(distribution, 1)
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
            a_loss = actor_loss(adv.detach(), distribution[a])
            a_loss.backward()
            a_optimizer.step()
                
            c_optimizer.zero_grad()
            c_loss = critic_loss(y, v_s_t)
            c_loss.backward()
            c_optimizer.step()
            
            if terminated:
                break
     
    torch.save(actor.state_dict(), "control/models/model.pth")
        

if __name__ == "__main__":
    dqn()
    env = gym.make("CartPole-v1", render_mode="human").unwrapped
    env.reset(seed=123, options={"low": -0.1, "high": 0.1})
    net = QNN()
    net.load_state_dict(torch.load("control/models/model_dqn.pth", weights_only=True))
    terminated = False
    r_sum = 0
    while not terminated:
        a = torch.argmax(net(preprocess_state(env.state))).item()
        s_t_1, r, terminated, _, _ = env.step(a)
        r_sum += r
    print(r_sum)
    
    '''
    a2c()
    env = gym.make("CartPole-v1", render_mode="human").unwrapped
    env.reset(seed=123, options={"low": -0.1, "high": 0.1})
    net = NNActor()
    net.load_state_dict(torch.load("control/models/model.pth", weights_only=True))
    terminated = False
    r_sum = 0
    while not terminated:
        distribution = net(preprocess_state(env.state))
        a = torch.multinomial(distribution, 1)
        s_t_1, r, terminated, _, _ = env.step(a.item())
        r_sum += r
    print(r_sum)
    '''
        
        
        

