import torch as th
import gym
from gym_minigrid.wrappers import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

policy_name = 'PPO'
# env_name = 'CartPole-v1'
env_name = 'MiniGrid-Empty-8x8-v0'
results_root = './rl_results/{}_{}'.format(policy_name, env_name)

# Custom actor (pi) and value function (vf) networks
# of two layers of size 32 each with Relu activation function
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[32, 32], vf=[32, 32])])
# Ceate the agent
# CartPole-v1:state->(1,4), action->(1,2)
if env_name in ['MiniGrid-Empty-8x8-v0']:
    print("good")
    env = gym.make(env_name)
    # check_env(env)
    env = RGBImgPartialObsWrapper(env)  # Get pixel observations
    env = ImgObsWrapper(env)  # Get rid of the 'mission' field
    model = PPO("CnnPolicy", env, learning_rate=1e-3, policy_kwargs=policy_kwargs,
                tensorboard_log="{}/tensorboard".format(results_root), verbose=1)
else:
    model = PPO("MlpPolicy", env_name, learning_rate=1e-3, policy_kwargs=policy_kwargs,
                tensorboard_log="{}/tensorboard".format(results_root), verbose=1)

# Train the agent
# Evaluate the model every 1000 steps on 5 test episodes
# and save the evaluation to the "logs/" folder
# total_timesteps:Number of interactions between agent and environment(one step==one transition);
# Each n_steps(2048) contains many episodes;
# Then n_steps transitions used to training.(1 epoch == n_steps transitions)
model.learn(total_timesteps=100000, eval_freq=1000, n_eval_episodes=5, eval_log_path="./logs/")
# save the model
model.save("{}/model".format(results_root))

# et policy
policy = model.policy
# Retrieve the environment
env = model.get_env()
# Evaluate the policy
mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# # load model

# del model
# # the policy_kwargs are automatically loaded
# model = PPO.load("ppo_cartpole", env=env)