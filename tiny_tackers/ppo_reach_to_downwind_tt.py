import os
import sys
import time
import builtins
import pandas as pd
import gymnasium as gym
import pygame

from stable_baselines3 import PPO

builtins.quit = lambda *args, **kwargs: None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO_ROOT)

GYM_SAIL_STEPS_PARENT = os.path.join(
    REPO_ROOT,
    "gym_sailing_environments",
    "gym_sail_steps",
)

GYM_SAIL_STEPS_PACKAGE = os.path.join(
    GYM_SAIL_STEPS_PARENT,
    "gym_sail_steps",
)

if not os.path.isdir(GYM_SAIL_STEPS_PACKAGE):
    raise FileNotFoundError(
        "Could not find the local gym_sail_steps package.\n"
        f"Expected package folder at: {GYM_SAIL_STEPS_PACKAGE}\n"
        f"Current working directory: {os.getcwd()}\n"
        "Confirm that the repo contains: "
        "gym_sailing_environments/gym_sail_steps/gym_sail_steps"
    )

sys.path.insert(0, GYM_SAIL_STEPS_PARENT)

# type ignore because the package is loaded by path above.
import gym_sail_steps  # type: ignore

print("Loaded gym_sail_steps from:", gym_sail_steps.__file__)

ENV_ID = "SailboatReachToDownwind-v0"
MAX_STEPS = 10_000
MODEL_PATH = "models/ppo/reach_to_downwind_step/ppo_upwind_1M_windwardtoreach_250K_reachtoleeward_250K.zip"
RESULTS_DIR = "data/results"
RESULTS_PATH = os.path.join(RESULTS_DIR, "ppo_reach_to_downwind_score.csv")


def run_ppo_episode():
    pygame.init()

    model = PPO.load(MODEL_PATH)
    env = gym.make(ENV_ID, render_mode="human")
    obs, info = env.reset()

    total_reward = 0.0
    timesteps = 0
    running = True

    print("PPO Reach to Downwind Model running...")
    print("ESC = quit episode")

    try:
        while running and timesteps < MAX_STEPS:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            timesteps += 1

            if terminated or truncated:
                break

            time.sleep(0.03)

    finally:
        try:
            env.close()
        except Exception:
            pass

        try:
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass

    return {
        "agent": "PPO Reach to Downwind Model",
        "env_id": ENV_ID,
        "model_path": MODEL_PATH,
        "total_reward": total_reward,
        "timesteps": timesteps,
    }


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    result = run_ppo_episode()
    pd.DataFrame([result]).to_csv(RESULTS_PATH, index=False)

    print("Saved PPO score:")
    print(result)