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
