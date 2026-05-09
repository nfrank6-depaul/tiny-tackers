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

GYM_RACE_PARENT = os.path.join(
    REPO_ROOT,
    "gym_sailing_environments",
    "gym_sailing_race",
)
GYM_RACE_PACKAGE = os.path.join(GYM_RACE_PARENT, "gym_sail_race")

if not os.path.isdir(GYM_RACE_PACKAGE):
    raise FileNotFoundError(
        "Could not find the local gym_sail_race package.\n"
        f"Expected package folder at: {GYM_RACE_PACKAGE}\n"
        f"Current working directory: {os.getcwd()}\n"
        "Confirm that the repo contains: "
        "gym_sailing_environments/gym_sailing_race/gym_sail_race"
    )

sys.path.insert(0, GYM_RACE_PARENT)

import gym_sail_race  # type: ignore

print("Loaded gym_sail_race from:", gym_sail_race.__file__)

ENV_ID = "SailboatRace-v0"
MAX_STEPS = 10_000

# Load the PPO model trained on the base Sailboat-v0 environment
MODEL_PATH = "models/ppo/base/ppo_base_1M.zip"

RESULTS_DIR = "data/results"
RESULTS_PATH = os.path.join(RESULTS_DIR, "ppo_base_to_race_score.csv")


def run_ppo_episode():
    pygame.init()

    model = PPO.load(MODEL_PATH)
    env = gym.make(ENV_ID, render_mode="human")

    obs, info = env.reset()

    total_reward = 0.0
    timesteps = 0
    running = True
    final_info = info

    print("PPO Base Model running on SailboatRace-v0...")
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
            final_info = info

            if timesteps % 100 == 0:
                print(
                    f"step={timesteps}, "
                    f"total_reward={total_reward:.2f}, "
                    f"reward={reward:.2f}, "
                    f"info={info}"
                )

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
        "agent": "PPO Base Model on SailboatRace-v0",
        "source_model": MODEL_PATH,
        "env_id": ENV_ID,
        "total_reward": total_reward,
        "timesteps": timesteps,
        "final_info": str(final_info),
    }


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    result = run_ppo_episode()
    pd.DataFrame([result]).to_csv(RESULTS_PATH, index=False)

    print("Saved PPO base-to-race score:")
    print(result)