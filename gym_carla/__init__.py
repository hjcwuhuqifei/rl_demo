from gym.envs.registration import register

register(
    id='carla-v1',
    entry_point='gym_carla.envs:CarlaEnv',
)


register(
    id='carla-v2',
    entry_point='gym_carla.envs:CarlaEnv2',
) 

register(
    id='carla-v11',
    entry_point='gym_carla.envs:CarlaEnv11',
)

register(
    id='carla-v21',
    entry_point='gym_carla.envs:CarlaEnv21',
)

register(
    id='carla-v_new',
    entry_point='gym_carla.envs:CarlaEnvNew',
)