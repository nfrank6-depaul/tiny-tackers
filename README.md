
# Tiny-Tackers

A reinforcement learning sailing environment and autonomous sailboat training project built with Python, Gymnasium, and Stable-Baselines3.

## Overview

Tiny-Tackers is a custom sailing simulation environment focused on autonomous sailboat navigation and racing behavior. The project explores curriculum learning / transfer learning by training with a Proximal Policy Optimization (PPO) Algorithm designed by Stable-Baselines3.  

 ### Phase 1: 
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

### Phase 2:
### Regardless of how well the PPO tiny-tacker performs, you (the human) must complete each of the learning tasks from phase 1.

- You will compete with the PPO for the lowest amount of timesteps needed to complete the learning tasks.


### Motivation for the project:
The project is inspired by real-world dinghy sailing, autonomous sailing research, and reinforcement learning literature.
 
# Environments
### gym_sail_steps: This is my set of environments based on Gabo-Tor's gym_sailing
(gym_sailing_environments/gym_sail_steps/gym_sail_steps)

The main differences included in my evironments were buoy placement, moving the goal position off the buoy itself, implementing reward curations per task, and altering the start and finish positions and requirements

Small adjustments were made to boat_env.py, the physics folder, and the renderer to accomodate video recording, graphics, and more realistic sailing, like the addition of a "dead downwind zone" that reduces speed when heading straight downwind.

The following environments were trained in the order you see them, but I was unsuccessful in geting the boat to jibe. 

- SailboatUpwind-v0 
    - Similar environment to Gabo Tor's baseline, only there is a more realistic target (called a rounding gate) that emmulates where you'd want to be positioned in a race
- SailboatWindwardToReach-v0 
    - This environment simulates the second leg of a triangle race, going from the windward buoy to the reach buoy
- SailboatReachToDownwind-v0 
    - This environment simulates the third leg of a traingle race, going from the reach buoy to the leeward buoy
- SailboatReachRounding-v0 
    - This environment attemps to force the agents to learn jibing behavior, currently, there is a large negative reward in this environment for turning upwind that has had little effect on helping the agent learn jibing behavior (this is where I need help from you!)
- SailboatDownwind-v0 
    - This environment simulates the downwind portion of a triangle race as we have already learned the upwind portion
- SailboatTriangle-v0 
    - This is the full triangle racing environment


---
Gabo-Tor's Original Repo: https://github.com/Gabo-Tor/gym-sailing
Their original license was preserved in this project as GABO-TOR_LICENSE_NOTES gym_sailing_environments/gym_sailing_gabo-tor

# Project Structure

```text
tiny-tackers/
│
├── data/                            # Where the results of human runs reside
├── gym_sailing_environments/        # Where my custom environments and Gabo-Tor's environment reside
├── metrics/                         # Where the results of your model training runs resides
├── models/                          # Where you can store saved trained PPO models
├── notebooks/                       # Model where model training and human training (Phase 1 and 2) occurs. Model training should be done in Colab
├── tiny_tackers/                    # Scripts that run pygame on your local device for both the PPO agent and the human
├── videos/                          # Where the videos of an evealuation run of specific models are stored
├── LICENSE                          # MIT License - based on the same license baseline environment maker (Gabo-Tor) used
├── requirements.txt                 # The packages used in this project that will need to be installed, can be done inside the notebooks as well
└── README.md
```

---

# Installation

## Clone Repository


git clone https://github.com/YOUR_USERNAME/tiny-tackers.git
cd tiny-tackers


## Create Virtual Environment

### macOS/Linux


python -m venv .venv
source .venv/bin/activate


## Install Dependencies

pip install -r requirements.txt

---

# Training/Video Recording/Human Learning and Evaluation

### These all occur in the two notebooks included in this project. 
- human_training.ipynb (run on your local machine using pygame :), see if you can beat the PPO and learn how to sail) 
- model_training.ipynb (use Colab)

# Reinforcement Learning Notes

This project experiments with:

- PPO (Proximal Policy Optimization)
- Curriculum learning
- Transfer learning
- Reward shaping
- Continuous action spaces

---

# Future Work

- GETTING THE AGENT TO JIBE (GYBE)
    - turning downwind proved to be something I could not get the agent to do reliably
- Add Dynamic wind shifts, create a stochastic environment variant
- Incorporate Multi-agent racing
- Institute sail trim controls
- Add Ocean currents
- Implement hierarchical decision-making to separate high-level sailing strategy from low-level control actions

---

# Research Inspiration

Relevant papers and resources include:

- Proximal Policy Optimization Algorithms http://arxiv.org/abs/1707.06347
- Trust Region Policy Optimization https://proceedings.mlr.press/v37/schulman15.html
- Stable-Baselines3 http://jmlr.org/papers/v22/20-1364.html
- AI-Based Autonomous Sailboat Navigation: A Review https://doi.org/10.1002/rob.70004

---

# License

MIT License

---

# Author

Nicole Frank

05/27/2026

MS Artificial Intelligence — DePaul University