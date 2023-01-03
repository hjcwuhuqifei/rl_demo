import argparse
import torch
import time
import os
import numpy as np
import gym
import carla
import gym_carla
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
# from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG

USE_CUDA = True  # torch.cuda.is_available()


# 运行MADDPG的主文件


def run(config_):
    writer = SummaryWriter("runs/" + "MADDPG100")

    # 生成环境的配置参数
    params = {
        'dt': 0.1,  # time interval between two frames
        'port': 2000,  # connection port
        'town': 'Town04',  # which town to simulate
        'max_time_episode': 1000,  # maximum timesteps per episode
        'punish_time_episode': 300,  # maximum timesteps per episode
        'desired_speed': 6,  # desired speed (m/s)
    }
    torch.manual_seed(config_.seed)
    np.random.seed(config_.seed)
    if not USE_CUDA:
        torch.set_num_threads(config_.n_training_threads)
    # 生成环境
    env = gym.make('carla-v_new', params=params)
    # 生成算法网络结构的参数
    agent_init_params = [{'num_in_pol': 12,
                          'num_out_pol': 1,
                          'num_in_critic': 39}, {'num_in_pol': 12,
                                                 'num_out_pol': 1,
                                                 'num_in_critic': 39}, {'num_in_pol': 12,
                                                                        'num_out_pol': 1,
                                                                        'num_in_critic': 39}]
    init_dict = {'gamma': 0.95, 'tau': config_.tau, 'lr': config_.lr,
                 'hidden_dim': config_.hidden_dim,
                 'alg_types': ['MADDPG', 'MADDPG', 'MADDPG'],
                 'agent_init_params': agent_init_params,
                 'discrete_action': False}
    # 生成算法
    maddpg = MADDPG(**init_dict)
    # 生成replaybuffer
    replay_buffer = ReplayBuffer(config_.buffer_length, maddpg.nagents,
                                 [12, 12, 12],
                                 [1, 1, 1])
    t = 0
    collision = 0
    success = 0
    frame = 0
    for ep_i in range(0, config_.n_episodes, config_.n_rollout_threads):
        # 开始迭代，初步设定周期为25000个
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config_.n_rollout_threads,
                                        config_.n_episodes))
        # 初始化环境
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config_.n_exploration_eps - ep_i) / config_.n_exploration_eps
        # 设置noise
        maddpg.scale_noise(
            config_.final_noise_scale + (config_.init_noise_scale - config_.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        for et_i in range(config_.episode_length):
            # 进入一个周期的训练
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = Variable(torch.Tensor(np.array(obs)),
                                 requires_grad=False)
            # get actions as torch Variables
            # 取得生成的actions
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config_.n_rollout_threads)]
            print(actions)
            # 环境运行一步
            next_obs, rewards, dones, infos = env.step(actions)
            # 得到的数据推入replaybuffer
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)

            frame += 1

            obs = next_obs
            t += config_.n_rollout_threads
            if (len(replay_buffer) >= config_.batch_size and
                    (t % config_.steps_per_update) < config_.n_rollout_threads):
                # 当replaybuffer内数据量大于一定程度时，开始训练
                print('start traning!!!')
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config_.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config_.batch_size,
                                                      to_gpu=USE_CUDA)
                        # 更新参数
                        maddpg.update(sample, a_i)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
            # 如果结束，退出此环境。以碰撞结束则将碰撞次数加一。
            if dones[0] and dones[1] and dones[2]:
                print('success!!!')
                success += 1
                break
            if dones[3]:
                collision += 1
                break
            if dones[4]:
                break
        if ep_i > 20:
            writer.add_scalar("MADDPG ego RL reward", rewards[0], frame)
            writer.add_scalar("MADDPG surround1 RL reward", rewards[1], frame)
            writer.add_scalar("MADDPG surround2 RL reward", rewards[2], frame)

        if ep_i % 100 == 0:
            writer.add_scalar("Suceess rate", success / 100, frame)
            writer.add_scalar("Collision Rate", collision / 100, frame)
            success = 0
            collision = 0


if __name__ == '__main__':
    # 生成算法的配置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default='sattr', type=str, help="Name of environment")
    parser.add_argument("--model_name", default='MADDPG', type=str,
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=10000, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=2, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')

    config = parser.parse_args()

    t0 = time.time()
    run(config)

    t1 = time.time()

    print("Training time: {}min".format(round((t1 - t0) / 60, 2)))
