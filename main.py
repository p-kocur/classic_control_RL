import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from algorithms.a2c import a2c
from algorithms.dqn import dqn, QNN
from algorithms.ppo import ppo, preprocess_state, NNActor, torch

if __name__ == "__main__":
    '''
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
    '''
    a2c()
    env = gym.make("CartPole-v1", render_mode="human").unwrapped
    env.reset(seed=123, options={"low": -0.1, "high": 0.1})
    net = NNActor()
    net.load_state_dict(torch.load("control/models/model_a2c.pth", weights_only=True))
    terminated = False
    r_sum = 0
    while not terminated:
        distribution = net(preprocess_state(env.state))
        a = torch.multinomial(distribution, 1)
        s_t_1, r, terminated, _, _ = env.step(a.item())
        r_sum += r
    print(r_sum)
    '''
    
    times_dqn, r_dqn = dqn()
    times_ppo, r_ppo = ppo()
    times_a2c, r_a2c = a2c() 
    
    
    
    fig, ax = plt.subplots()
    ax.title.set_text("Results")
    ax.plot(np.array(times_dqn), np.array(r_dqn), label="DQN")
    ax.plot(np.array(times_a2c), np.array(r_a2c), label="A2C")
    ax.plot(np.array(times_ppo), np.array(r_ppo), label="PPO")
    ax.legend()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Reward")
    ax.grid(True)
    plt.show()
    
    
    fig.savefig("figures/results.png")
    
    
        
        

