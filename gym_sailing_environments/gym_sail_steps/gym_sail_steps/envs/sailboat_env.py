import numpy as np

from gym_sail_steps.envs.boat_env import BoatDiscreteEnv, BoatEnv
from gym_sail_steps.physics.sailboat import SailBoat


class SailboatEnvUpwind(BoatEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)

    def reset(self, options=None, seed=None):
        self.boat = SailBoat(
            x=self.COURSE_SIZE * (0.5 + np.random.uniform(-0.2, 0.2)),
            y=self.COURSE_SIZE * 0.10,
            heading=self.np_random.random() * np.pi * 2,
            heading_dot=np.random.uniform(-0.03, 0.03),
            speed=np.random.uniform(-1, 0.5),
        )
        return super().reset(options, seed)

class SailboatEnvDownwind(BoatEnv):
    TARGET = (BoatEnv.COURSE_SIZE * 0.5, BoatEnv.COURSE_SIZE * 0.10)

    def __init__(self, render_mode=None):
        super().__init__(render_mode)

    def reset(self, options=None, seed=None):
        self.boat = SailBoat(
            x=self.COURSE_SIZE * (0.5 + np.random.uniform(-0.2, 0.2)),
            y=self.COURSE_SIZE * 0.90,
            heading=-np.pi / 2 + np.random.uniform(-0.75, 0.75),
            heading_dot=np.random.uniform(-0.03, 0.03),
            speed=np.random.uniform(0, 0.5),
        )

        return super().reset(options, seed)
    
class SailboatEnvReach(BoatEnv):
    # Reach buoy: left side of the course, middle height
    TARGET = (BoatEnv.COURSE_SIZE * 0.20, BoatEnv.COURSE_SIZE * 0.50)
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
    def reset(self, options=None, seed=None):
        target_y = self.TARGET[1]
        self.boat = SailBoat(
            # Start on the right side
            x=self.COURSE_SIZE * np.random.uniform(0.75, 0.90),
            # Sometimes slightly upwind, sometimes slightly downwind of the reach buoy
            y=target_y + self.COURSE_SIZE * np.random.uniform(-0.15, 0.15),
            # Start roughly aimed leftward toward the reach buoy
            heading=np.pi + np.random.uniform(-0.75, 0.75),
            heading_dot=np.random.uniform(-0.03, 0.03),
            speed=np.random.uniform(0, 0.5),
        )
        return super().reset(options, seed)


class SailboatEnvTriangle(BoatEnv):
    WINDWARD_BUOY = (BoatEnv.COURSE_SIZE * 0.50, BoatEnv.COURSE_SIZE * 0.90)
    REACH_BUOY = (BoatEnv.COURSE_SIZE * 0.20, BoatEnv.COURSE_SIZE * 0.50)
    LEEWARD_BUOY = (BoatEnv.COURSE_SIZE * 0.50, BoatEnv.COURSE_SIZE * 0.10)

    COURSE = [
        WINDWARD_BUOY,
        REACH_BUOY,
        LEEWARD_BUOY,
        WINDWARD_BUOY,
        LEEWARD_BUOY,
        WINDWARD_BUOY,
    ]

    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.current_mark_index = 0
        self.TARGET = self.COURSE[self.current_mark_index]

    def reset(self, options=None, seed=None):
        self.current_mark_index = 0
        self.TARGET = self.COURSE[self.current_mark_index]

        leeward_x, leeward_y = self.LEEWARD_BUOY

        self.boat = SailBoat(
            # A little right of the leeward/downwind buoy
            x=leeward_x + self.COURSE_SIZE * np.random.uniform(0.05, 0.20),

            # A little downwind of the leeward/downwind buoy
            y=leeward_y - self.COURSE_SIZE * np.random.uniform(0.03, 0.10),

            # Roughly pointed toward the windward buoy
            heading=np.pi / 2 + np.random.uniform(-0.75, 0.75),

            heading_dot=np.random.uniform(-0.03, 0.03),
            speed=np.random.uniform(0, 0.5),
        )

        return super().reset(options, seed)

    def _get_reward(self, distance2target):
        terminated = False
        reward = -0.1

        distance = np.linalg.norm(distance2target)

        # Reached current buoy
        if distance < self.TARGET_RAD:
            self.current_mark_index += 1

            # Finished all marks
            if self.current_mark_index >= len(self.COURSE):
                reward += 300
                terminated = True
                self.last_reward = reward
                return terminated, reward

            # Move to next buoy
            self.TARGET = self.COURSE[self.current_mark_index]

            # Reward reaching an intermediate buoy
            reward += 75

            # Reset previous distance for new target
            self.prev_distance2target = np.array([self.boat.x, self.boat.y]) - np.array(
                self.TARGET
            )

            self.last_reward = reward
            return terminated, reward

        # Out of bounds / too far away
        if distance >= self.COURSE_SIZE:
            reward = -100
            terminated = True

        # Progress toward current buoy
        reward += 10 * (
            np.linalg.norm(self.prev_distance2target, 8)
            - np.linalg.norm(distance2target, 8)
        )

        self.prev_distance2target = distance2target
        self.last_reward = reward
        return terminated, reward

class SailboatDiscreteEnv(BoatDiscreteEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)

    def reset(self, options=None, seed=None):
        self.boat = SailBoat(
            x=self.COURSE_SIZE * (0.5 + np.random.uniform(-0.2, 0.2)),
            y=self.COURSE_SIZE * 0.10,
            heading=self.np_random.random() * np.pi * 2,
            heading_dot=np.random.uniform(-0.03, 0.03),
            speed=np.random.uniform(-1, 0.5),
        )
        return super().reset(options, seed)
