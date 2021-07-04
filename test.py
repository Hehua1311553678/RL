import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

results_root = './results/ppo_cartpole'
env_name = 'CartPole-v1'

# Create environment
env = gym.make(env_name)

model = PPO.load("{}/ppo_cartpole".format(results_root), env=env)

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()