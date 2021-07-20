import os
import pickle
import numpy as np

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.policies import serialize
from imitation.algorithms import bc

env_name = 'CartPole-v1'

# Create environment
env = gym.make(env_name)

# 1. rl policy
# results_root = './results/ppo_cartpole'
# model = PPO.load("{}/ppo_cartpole".format(results_root), env=env)

# 2. expert policy
# policy_path = './expert/ppo_CartPole-v1/policies/final'
# with open(os.path.join(policy_path, "vec_normalize.pkl"), "rb") as f:
#     venv = pickle.load(f)
# model = serialize.load_policy(policy_type='ppo', policy_path=policy_path, venv=venv)

# 3. BC policy
# policy_path = './il_results/ppo_CartPole-v1/BC/BC_policy.pth.tar'
# model = bc.reconstruct_policy(policy_path)

# 4. GAIL policy
policy_path = './il_results/ppo_CartPole-v1/GAIL/GAIL_policy.pth.tar'
model = bc.reconstruct_policy(policy_path)

# Evaluate the policy
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
# print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# episode_rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True, return_episode_rewards=True)
# mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
# print("episode_rewards={} \nepisode_lengths={}".format(episode_rewards, episode_lengths))
# print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
# print("len={}, set_episode_rewards={}".format(len(episode_rewards), set(episode_rewards)))

# Enjoy trained agent
obs = env.reset()
for i in range(500):
    print("i={}".format(i))
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()

