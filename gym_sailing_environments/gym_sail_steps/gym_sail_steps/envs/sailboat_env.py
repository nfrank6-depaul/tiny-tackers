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

    UPWIND_HEADING_PENALTY = 2.0
    SPIN_PENALTY = 5.0

    def __init__(self, render_mode=None):
        super().__init__(render_mode)

    def _angle_diff(self, a, b):
        return (a - b + np.pi) % (2 * np.pi) - np.pi

    def _is_near_upwind(self):
        return abs(self._angle_diff(self.boat.heading, np.pi / 2)) < 0.6

    def _get_reward(self, distance2target):
        terminated, reward = super()._get_reward(distance2target)

        if self._is_near_upwind():
            reward -= self.UPWIND_HEADING_PENALTY

        if abs(self.boat.heading_dot) > 0.25:
            reward -= self.SPIN_PENALTY

        self.last_reward = reward
        return terminated, reward

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
                COURSE_CENTER_X + np.random.uniform(-0.15, 0.15)
            ),
            y=self.COURSE_SIZE * WINDWARD_Y,
            heading=-np.pi / 2 + np.random.uniform(-0.75, 0.75),
            heading_dot=np.random.uniform(-0.03, 0.03),
            speed=np.random.uniform(0, 0.5),
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


