import torch
import argparse
from pathlib import Path
import numpy as np
# input = torch.rand(3,16,16,dtype=torch.float32)
#
# Linear = torch.nn.Linear(16,322)
#
# conv = torch.nn.Conv2d(3,10,kernel_size=(3,3),stride=1,padding=1)
#
# # output = Linear(input)
#
# print(input.shape)
# # print(output.shape)
#
# output = conv(input)
# print(output.shape)

# parser = argparse.ArgumentParser()
# parser.add_argument("--env_id", default='sattr',type=str, help="Name of environment")
# parser.add_argument("--model_name", default='MADDPG', type=str,
#                     help="Name of directory to store " +
#                          "model/training contents")
# parser.add_argument("--seed",
#                         default=1, type=int,
#                         help="Random seed")
# parser.add_argument("--n_rollout_threads", default=1, type=int)
# parser.add_argument("--n_training_threads", default=6, type=int)
# parser.add_argument("--buffer_length", default=int(1e6), type=int)
# parser.add_argument("--n_episodes", default=25000, type=int)
# parser.add_argument("--episode_length", default=25, type=int)
# parser.add_argument("--steps_per_update", default=100, type=int)
# parser.add_argument("--batch_size",
#                     default=1024, type=int,
#                     help="Batch size for model training")
# parser.add_argument("--n_exploration_eps", default=25000, type=int)
# parser.add_argument("--init_noise_scale", default=0.3, type=float)
# parser.add_argument("--final_noise_scale", default=0.0, type=float)
# parser.add_argument("--save_interval", default=1000, type=int)
# parser.add_argument("--hidden_dim", default=64, type=int)
# parser.add_argument("--lr", default=0.01, type=float)
# parser.add_argument("--tau", default=0.01, type=float)
# parser.add_argument("--agent_alg",
#                     default="MADDPG", type=str,
#                     choices=['MADDPG', 'DDPG'])
# parser.add_argument("--adversary_alg",
#                     default="MADDPG", type=str,
#                     choices=['MADDPG', 'DDPG'])
# parser.add_argument("--discrete_action",
#                     action='store_true')
# config = parser.parse_args()
# model_dir = Path('./models') / config.env_id / config.model_name
# print(parser.parse_args())
# print(parser.parse_args().lr)
# print(model_dir)

A = np.array([[1,2,3], [1,2,3], [1,2,3]])
b = np.vstack(A[:, 2])
print(A)
print(A[:, 2])