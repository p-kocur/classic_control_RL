# Testing RL algorithms (DQN, A2C, PPO) in CartPole enviroment

This repository contains implementations and tests of RL algorithms: Deep Q-Learning (DQN), Advantage Actor-Critic (A2C) and Proximal Policy Optimization (PPO) in `CartPole` enviroment from OpenAI Gym.

## Requirements

Minimal Python version: 3.7.

1. Copy the repository:
   ```bash
   git clone https://github.com/p-kocur/classic_control_RL
   cd classic_control_RL
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## How to use

### Testing algorithms

```bash
python main.py
```

### Observe trained policies in action

```bash
python see_results.py -o [a2c, ppo, dqn]
```
Replace `[a2c, ppo, dqn]` with algorithm name, which you want to use:
```bash
python see_results.py -o ppo
```

## Project structure

- `main.py` - Main script
- `see_results.py` - Script for visualization
- `requirements.txt` -Required libraries list
- `models/` - Folder containing saved models
- `algorithms/` - Folder containing training algorithms
- `figures/` - Folder containing plot figure of last experiment

## Author
Paweł Kocur
