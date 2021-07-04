# 现有强化学习方法
## 一、基本概念
### 1. 问题设定
智能决策问题/序贯决策问题（需要连续做出决策才能实现最终目标的问题）。
### 2. 问题
（1）model-free RL的sample inefficient：
有时候训练需要数百万次交互，所以我们训练的时候最好增大agent的budget（训练时间步数）；

（2）reward function的设计：
reward shaping的一个好的例子：DeepMimic（将Imitation Learning与RL相结合来学习机器人移动的行为）;

（3）训练的不稳定性（instability of training）：训练过程中观察到性能的大幅下降
（DDPG这种情况特别明显，TD3作为DDPG的扩展解决的就是不稳定性这个问题，
而TRPO与PPO通过trust region避免过大的更新从而最小化训练不稳定的问题）

### 3. 其他
（1）DQN只支持离散动作，SAC只支持连续动作；

（2）DQN训练缓慢，但是most sample efficient（因为有replay buffer）；

（3）normalization、batch size

（4）RL环境：

- 连续动作：Pendulum（易）、HalfCheetahBullet（中）、BipedalWalkerHardcore（难）；

- 离散动作：CartPole-v1（较易比随机agent好，但是很难获得最好的性能）、LunarLander、Pong（Atari game最简单的游戏之一）、
  其他的Atari games (如Breakout)
  
## 二、算法
### 1. PPO（Proximal Policy Optimization algorithm）
(1)on-policy 算法;

(2)动作空间: discrete/continuous;

