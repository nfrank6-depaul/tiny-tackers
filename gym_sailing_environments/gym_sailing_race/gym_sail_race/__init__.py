from gymnasium.envs.registration import register

register(
    id="SailboatRace-v0",
    entry_point="gym_sail_race.envs:SailboatRaceEnv",
    max_episode_steps=20000,
)
register(
    id="SailboatRaceJibePractice-v0",
    entry_point="gym_sail_race.envs:SailboatRaceEnvJibePractice",
    max_episode_steps=20000,
)

register(
    id="SailboatRace2Mark-v0",
    entry_point="gym_sail_race.envs.sailboat_env:SailboatRaceEnv2Mark",
    max_episode_steps=20000,
)

register(
    id="SailboatDiscrete-v0",
    entry_point="gym_sail_race.envs:SailboatDiscreteEnv",
    max_episode_steps=3000,
)

register(
    id="Motorboat-v0",
    entry_point="gym_sail_race.envs:MotorboatEnv",
    max_episode_steps=2000,
)
