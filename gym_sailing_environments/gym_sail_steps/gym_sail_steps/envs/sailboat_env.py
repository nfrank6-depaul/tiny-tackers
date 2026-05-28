import numpy as np

from gym_sail_steps.envs.boat_env import BoatEnv
from gym_sail_steps.physics.sailboat import SailBoat

COURSE_CENTER_X = 0.62
REACH_X = 0.20
WINDWARD_Y = 0.85
REACH_Y = 0.50
LEEWARD_Y = 0.15

class SailboatEnvUpwind(BoatEnv):
    WINDWARD_BUOY = (BoatEnv.COURSE_SIZE * COURSE_CENTER_X, BoatEnv.COURSE_SIZE * WINDWARD_Y)
    ROUNDING_GATE = (
        BoatEnv.COURSE_SIZE * (COURSE_CENTER_X + 0.025),
        BoatEnv.COURSE_SIZE * (WINDWARD_Y + 0.015),
    )
    TARGET = ROUNDING_GATE
    def __init__(self, render_mode=None):
        super().__init__(render_mode)

    def _render_frame(self):
        return self.renderer._render_frame(
            boats=[
                (
                    self.boat.x,
                    self.boat.y,
                    self.boat.heading - np.pi / 2,
                    self.last_action,
                )
            ],
            target=self.WINDWARD_BUOY,
            gate=self.ROUNDING_GATE,
            stepnum=self.stepnum,
            reward=self.last_reward,
            render_mode=self.render_mode,
            fps=self.metadata["render_fps"],
        )

    def reset(self, options=None, seed=None):
        self.boat = SailBoat(
            x=self.COURSE_SIZE * (COURSE_CENTER_X + np.random.uniform(-0.15, 0.15)),
            y=self.COURSE_SIZE * LEEWARD_Y,
            heading=self.np_random.random() * np.pi * 2,
            heading_dot=np.random.uniform(-0.03, 0.03),
            speed=np.random.uniform(-1, 0.5),
        )
        return super().reset(options, seed)
    
    

class SailboatEnvDownwind(BoatEnv):
    LEEWARD_BUOY = (
        BoatEnv.COURSE_SIZE * COURSE_CENTER_X,
        BoatEnv.COURSE_SIZE * LEEWARD_Y,
    )

    ROUNDING_GATE = (
        BoatEnv.COURSE_SIZE * (COURSE_CENTER_X - 0.025),
        BoatEnv.COURSE_SIZE * (LEEWARD_Y - 0.015),
    )

    TARGET = ROUNDING_GATE

    def __init__(self, render_mode=None):
        super().__init__(render_mode)

    def _render_frame(self):
        return self.renderer._render_frame(
            boats=[
                (
                    self.boat.x,
                    self.boat.y,
                    self.boat.heading - np.pi / 2,
                    self.last_action,
                )
            ],
            target=self.LEEWARD_BUOY,
            gate=self.ROUNDING_GATE,
            stepnum=self.stepnum,
            reward=self.last_reward,
            render_mode=self.render_mode,
            fps=self.metadata["render_fps"],
        )

    def reset(self, options=None, seed=None):
        self.boat = SailBoat(
            x=self.COURSE_SIZE * (
                COURSE_CENTER_X + self.np_random.uniform(-0.15, 0.15)
            ),
            y=self.COURSE_SIZE * WINDWARD_Y,
            heading=3 * np.pi / 2 + self.np_random.uniform(-0.75, 0.75),
            heading_dot=self.np_random.uniform(-0.03, 0.03),
            speed=self.np_random.uniform(0, 0.5),
        )

        return super().reset(options, seed)
    
class SailboatEnvWindwardToReach(BoatEnv):
    WINDWARD_BUOY = (
        BoatEnv.COURSE_SIZE * COURSE_CENTER_X,
        BoatEnv.COURSE_SIZE * WINDWARD_Y,
    )

    REACH_BUOY = (
        BoatEnv.COURSE_SIZE * REACH_X,
        BoatEnv.COURSE_SIZE * REACH_Y,
    )

    ROUNDING_GATE = (
        BoatEnv.COURSE_SIZE * (REACH_X - 0.025),
        BoatEnv.COURSE_SIZE * (REACH_Y + 0.015),
    )

    TARGET = ROUNDING_GATE

    def __init__(self, render_mode=None):
        super().__init__(render_mode)

    def _render_frame(self):
        return self.renderer._render_frame(
            boats=[
                (
                    self.boat.x,
                    self.boat.y,
                    self.boat.heading - np.pi / 2,
                    self.last_action,
                )
            ],
            target=self.REACH_BUOY,
            gate=self.ROUNDING_GATE,
            stepnum=self.stepnum,
            reward=self.last_reward,
            render_mode=self.render_mode,
            fps=self.metadata["render_fps"],
        )

    def reset(self, options=None, seed=None):
        windward_x, windward_y = self.WINDWARD_BUOY

        self.boat = SailBoat(
            # Start near windward mark
            x=windward_x + self.COURSE_SIZE * np.random.uniform(-0.08, 0.08),

            # Slightly below windward mark
            y=windward_y - self.COURSE_SIZE * np.random.uniform(0.03, 0.10),

            # Roughly pointed toward reach mark
            heading=np.pi + np.random.uniform(-0.75, 0.75),

            heading_dot=np.random.uniform(-0.03, 0.03),
            speed=np.random.uniform(0, 0.5),
        )

        return super().reset(options, seed)
    
class SailboatEnvReachToDownwind(BoatEnv):
    REACH_BUOY = (
        BoatEnv.COURSE_SIZE * REACH_X,
        BoatEnv.COURSE_SIZE * REACH_Y,
    )

    LEEWARD_BUOY = (
        BoatEnv.COURSE_SIZE * COURSE_CENTER_X,
        BoatEnv.COURSE_SIZE * LEEWARD_Y,
    )

    ROUNDING_GATE = (
        BoatEnv.COURSE_SIZE * (COURSE_CENTER_X - 0.025),
        BoatEnv.COURSE_SIZE * (LEEWARD_Y - 0.015),
    )

    TARGET = ROUNDING_GATE

    def __init__(self, render_mode=None):
        super().__init__(render_mode)

    def _render_frame(self):
        return self.renderer._render_frame(
            boats=[
                (
                    self.boat.x,
                    self.boat.y,
                    self.boat.heading - np.pi / 2,
                    self.last_action,
                )
            ],
            target=self.LEEWARD_BUOY,
            gate=self.ROUNDING_GATE,
            stepnum=self.stepnum,
            reward=self.last_reward,
            render_mode=self.render_mode,
            fps=self.metadata["render_fps"],
        )

    def reset(self, options=None, seed=None):
        reach_x, reach_y = self.REACH_BUOY

        self.boat = SailBoat(
            # Start near the reach mark
            x=reach_x + self.COURSE_SIZE * np.random.uniform(-0.08, 0.08),

            # Slightly above/below the reach mark
            y=reach_y + self.COURSE_SIZE * np.random.uniform(-0.08, 0.08),

            # Roughly pointed toward the leeward/downwind mark
            heading=-np.pi / 4 + np.random.uniform(-0.75, 0.75),

            heading_dot=np.random.uniform(-0.03, 0.03),
            speed=np.random.uniform(0, 0.5),
        )

        return super().reset(options, seed)


class SailboatEnvTriangle(BoatEnv):
    WINDWARD_BUOY = (
        BoatEnv.COURSE_SIZE * COURSE_CENTER_X,
        BoatEnv.COURSE_SIZE * WINDWARD_Y,
    )

    REACH_BUOY = (
        BoatEnv.COURSE_SIZE * REACH_X,
        BoatEnv.COURSE_SIZE * REACH_Y,
    )

    LEEWARD_BUOY = (
        BoatEnv.COURSE_SIZE * COURSE_CENTER_X,
        BoatEnv.COURSE_SIZE * LEEWARD_Y,
    )

    # Reward gates
    WINDWARD_GATE = (
        BoatEnv.COURSE_SIZE * (COURSE_CENTER_X + 0.025),
        BoatEnv.COURSE_SIZE * (WINDWARD_Y + 0.015),
    )

    REACH_GATE = (
        BoatEnv.COURSE_SIZE * (REACH_X - 0.025),
        BoatEnv.COURSE_SIZE * (REACH_Y + 0.015),
    )

    LEEWARD_GATE = (
        BoatEnv.COURSE_SIZE * (COURSE_CENTER_X - 0.025),
        BoatEnv.COURSE_SIZE * (LEEWARD_Y - 0.015),
    )

    # Visible buoys
    COURSE_BUOYS = [
        WINDWARD_BUOY,
        REACH_BUOY,
        LEEWARD_BUOY,
        WINDWARD_BUOY,
        LEEWARD_BUOY,
        WINDWARD_BUOY,
    ]

    # Hidden reward gates
    COURSE_GATES = [
        WINDWARD_GATE,
        REACH_GATE,
        LEEWARD_GATE,
        WINDWARD_GATE,
        LEEWARD_GATE,
        WINDWARD_GATE,
    ]

    def __init__(self, render_mode=None):
        super().__init__(render_mode)

        self.current_mark_index = 0

        self.CURRENT_BUOY = self.COURSE_BUOYS[self.current_mark_index]
        self.TARGET = self.COURSE_GATES[self.current_mark_index]

    def _render_frame(self):
        return self.renderer._render_frame(
            boats=[
                (
                    self.boat.x,
                    self.boat.y,
                    self.boat.heading - np.pi / 2,
                    self.last_action,
                )
            ],
            target=[self.WINDWARD_BUOY, self.REACH_BUOY, self.LEEWARD_BUOY],
            gate=self.TARGET,
            stepnum=self.stepnum,
            reward=self.last_reward,
            render_mode=self.render_mode,
            fps=self.metadata["render_fps"],
        )

    def reset(self, options=None, seed=None):
        self.current_mark_index = 0

        self.CURRENT_BUOY = self.COURSE_BUOYS[self.current_mark_index]
        self.TARGET = self.COURSE_GATES[self.current_mark_index]

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

        # Reached current gate
        if distance < self.TARGET_RAD:
            self.current_mark_index += 1

            # Finished race
            if self.current_mark_index >= len(self.COURSE_GATES):
                reward += 300
                terminated = True
                self.last_reward = reward
                return terminated, reward

            # Advance to next buoy/gate
            self.CURRENT_BUOY = self.COURSE_BUOYS[self.current_mark_index]
            self.TARGET = self.COURSE_GATES[self.current_mark_index]

            # Intermediate mark reward
            reward += 75

            # Reset progress tracking
            self.prev_distance2target = (
                np.array([self.boat.x, self.boat.y]) - np.array(self.TARGET)
            )

            self.last_reward = reward
            return terminated, reward

        # Out of bounds
        if distance >= self.COURSE_SIZE:
            reward = -100
            terminated = True

        # Progress reward
        reward += 10 * (
            np.linalg.norm(self.prev_distance2target, 8)
            - np.linalg.norm(distance2target, 8)
        )

        self.prev_distance2target = distance2target
        self.last_reward = reward

        return terminated, reward

class SailboatEnvReachRounding(BoatEnv):
    WINDWARD_BUOY = (
        BoatEnv.COURSE_SIZE * COURSE_CENTER_X,
        BoatEnv.COURSE_SIZE * WINDWARD_Y,
    )

    REACH_BUOY = (
        BoatEnv.COURSE_SIZE * REACH_X,
        BoatEnv.COURSE_SIZE * REACH_Y,
    )

    LEEWARD_BUOY = (
        BoatEnv.COURSE_SIZE * COURSE_CENTER_X,
        BoatEnv.COURSE_SIZE * LEEWARD_Y,
    )

    REACH_GATE = (
        BoatEnv.COURSE_SIZE * (REACH_X - 0.025),
        BoatEnv.COURSE_SIZE * (REACH_Y + 0.015),
    )

    LEEWARD_GATE = (
        BoatEnv.COURSE_SIZE * (COURSE_CENTER_X - 0.025),
        BoatEnv.COURSE_SIZE * (LEEWARD_Y - 0.015),
    )

    COURSE_BUOYS = [
        REACH_BUOY,
        LEEWARD_BUOY,
    ]

    TARGET_SEQUENCE = [
        REACH_GATE,
        LEEWARD_GATE,
    ]

    CLOCKWISE_TURN_PENALTY = -1
    ALIVE_PENALTY = -0.3
    INTERMEDIATE_GATE_REWARD = 75
    FINAL_GATE_REWARD = 300

    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.current_gate_index = 0
        self.TARGET = self.TARGET_SEQUENCE[self.current_gate_index]

    def _angle_diff(self, a, b):
        return (a - b + np.pi) % (2 * np.pi) - np.pi

    def step(self, action):
        previous_heading = self.boat.heading

        obs, reward, terminated, truncated, info = super().step(action)

        current_heading = self.boat.heading
        heading_change = self._angle_diff(current_heading, previous_heading)

        if heading_change < 0:
            reward += self.CLOCKWISE_TURN_PENALTY

        self.last_reward = reward

        return obs, reward, terminated, truncated, info

    def _render_frame(self):
        return self.renderer._render_frame(
            boats=[
                (
                    self.boat.x,
                    self.boat.y,
                    self.boat.heading - np.pi / 2,
                    self.last_action,
                )
            ],
            target=[
                self.REACH_BUOY,
                self.LEEWARD_BUOY,
            ],
            gate=self.TARGET,
            stepnum=self.stepnum,
            reward=self.last_reward,
            render_mode=self.render_mode,
            fps=self.metadata["render_fps"],
        )

    def reset(self, options=None, seed=None):
        self.current_gate_index = 0
        self.TARGET = self.TARGET_SEQUENCE[self.current_gate_index]

        windward_x, windward_y = self.WINDWARD_BUOY

        self.boat = SailBoat(
            x=windward_x + self.COURSE_SIZE * self.np_random.uniform(-0.08, 0.08),
            y=windward_y - self.COURSE_SIZE * self.np_random.uniform(0.03, 0.10),
            heading=np.pi + self.np_random.uniform(-0.75, 0.75),
            heading_dot=self.np_random.uniform(-0.03, 0.03),
            speed=self.np_random.uniform(0, 0.5),
        )

        return super().reset(options, seed)

    def _get_reward(self, distance2target):
        terminated = False
        reward = self.ALIVE_PENALTY

        distance = np.linalg.norm(distance2target)

        if distance < self.TARGET_RAD:
            self.current_gate_index += 1

            if self.current_gate_index >= len(self.TARGET_SEQUENCE):
                reward += self.FINAL_GATE_REWARD
                terminated = True
                self.last_reward = reward
                return terminated, reward

            self.TARGET = self.TARGET_SEQUENCE[self.current_gate_index]
            reward += self.INTERMEDIATE_GATE_REWARD

            self.prev_distance2target = (
                np.array([self.boat.x, self.boat.y]) - np.array(self.TARGET)
            )

            self.last_reward = reward
            return terminated, reward

        if distance >= self.COURSE_SIZE:
            reward = -100
            terminated = True

        reward += 10 * (
            np.linalg.norm(self.prev_distance2target, 8)
            - np.linalg.norm(distance2target, 8)
        )

        self.prev_distance2target = distance2target
        self.last_reward = reward

        return terminated, reward