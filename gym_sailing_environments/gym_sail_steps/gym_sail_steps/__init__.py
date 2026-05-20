from gymnasium.envs.registration import register

register(
    id="SailboatUpwind-v0",
    entry_point="gym_sail_steps.envs:SailboatEnvUpwind",
    max_episode_steps=10000,
)

register(
    id="SailboatDownwind-v0",
    entry_point="gym_sail_steps.envs:SailboatEnvDownwind",
    max_episode_steps=10000,
)

register(
    id="SailboatReach-v0",
    entry_point="gym_sail_steps.envs:SailboatEnvReach",
    max_episode_steps=10000,
)

register(
    id="SailboatTriangle-v0",
    entry_point="gym_sail_steps.envs:SailboatEnvTriangle",
    max_episode_steps=10000,
)

register(
    id="SailboatDiscrete-v0",
    entry_point="gym_sail_steps.envs:SailboatDiscreteEnv",
    max_episode_steps=10000,
)

register(
    id="Motorboat-v0",
    entry_point="gym_sail_steps.envs:MotorboatEnv",
    max_episode_steps=10000,
)
