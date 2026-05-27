
# Tiny-Tackers

A reinforcement learning sailing environment and autonomous sailboat training project built with Python, Gymnasium, and Stable-Baselines3.

## Overview

Tiny-Tackers is a custom sailing simulation environment focused on autonomous sailboat navigation and racing behavior. The project explores curriculum learning / transfer learning by training with a Proximal Policy Optimization (PPO) Algorithm designed by Stable-Baselines3.  

 ## Phase 1: 
 ### The objective is to train a PPO in sequence with the following learning tasks in order: 

- Upwind sailing
- Sailing from a windward buoy to a reach buoy
- Sailing from a reach buoy to a leeward buoy
- Downwind sailing
- Olympic Triangle Sailing Race Course Completion

Along the way the PPO tiny tacker (agent) should learn: 
- Tack and jibe behavior
- Buoy rounding
- Wind-aware navigation
- Long term strategizing

## Phase 2:
### Regardless of how well the PPO tiny-tacker performs, you (the human) must complete each of the learning tasks from phase 1.

- You will compete with the PPO for the lowest amount of timesteps needed to complete the learning tasks.


## Motivation
The project is inspired by real-world dinghy sailing, autonomous sailing research, and reinforcement learning literature.
 
  


---

# Features

- Custom Gymnasium-compatible sailing environments
- Realistic point-of-sail speed modeling
- Wind direction and no-go zones
- PPO training with Stable-Baselines3
- Video recording and evaluation tools
- Curriculum learning experiments
- Transfer learning between sailing environments
- Visualization and debugging utilities

---

# Project Structure

```text
tiny-tackers/
│
├── data/                            # Custom Gymnasium sailing environments
├── gym_sailing_environments/        # Saved trained RL models
├── metrics/                         # Evaluation videos
├── models/                          # Jupyter/Colab notebooks
├── notebooks/                       # Training scripts
├── tiny_tackers/                    # Evaluation scripts
├── videos/                          # Images and graphics
├── LICENSE                          # MIT License - based on the same license baseline environment maker (Gabo-Tor) used
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/tiny-tackers.git
cd tiny-tackers
```

## Create Virtual Environment

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

Main libraries used:

- gymnasium
- stable-baselines3
- torch
- numpy
- pygame
- matplotlib

---

# Training an Agent

Example PPO training:

```bash
python training/train_ppo.py
```

Or from a notebook:

```python
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
```

---

# Recording Evaluation Videos

Example:

```python
from gymnasium.wrappers import RecordVideo
```

Evaluation videos are saved to:

```text
/videos
```

---

# Reinforcement Learning Approaches

This project experiments with:

- PPO (Proximal Policy Optimization)
- DQN (Deep Q-Networks)
- Curriculum learning
- Transfer learning
- Reward shaping
- Continuous vs discrete action spaces

---

# Future Goals

- Dynamic wind shifts
- Multi-agent racing
- Sail trim controls
- Ocean current simulation
- Visual observation support
- Domain randomization
- Sim-to-real transfer concepts

---

# Research Inspiration

Relevant papers and resources include:

- Proximal Policy Optimization Algorithms
- Trust Region Policy Optimization
- Stable-Baselines3
- AI-Based Autonomous Sailboat Navigation: A Review

---

# License

MIT License

---

# Author

Nikki Frank

MS Artificial Intelligence — DePaul University