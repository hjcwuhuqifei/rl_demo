import torch
import gym
import carla
import gym_carla

# 测试车辆转弯的代码
params = {
    'dt': 0.1,  # time interval between two frames
    'port': 2000,  # connection port
    'town': 'Town04',  # which town to simulate
    'max_time_episode': 300,  # maximum timesteps per episode
    'desired_speed': 6,  # desired speed (m/s)
}

env = gym.make('carla-v_new', params=params)

# while True:
#     state = env.reset()
#     for i in range(800):
#         next_state, reward, done, _ = env.step(9)

