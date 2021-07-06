"""Loads CartPole-v1 demonstrations and trains BC, GAIL, and AIRL models on that data.
"""

import pathlib
import pickle
import tempfile

import stable_baselines3 as sb3

from imitation.algorithms import adversarial, bc
from imitation.data import rollout
from imitation.util import logger, util

def imitation_learning(expert_traj_path, imitation_algo_name, rl_algo_name, env_name):
    # Load pickled expert demonstrations.
    with open(expert_traj_path, "rb") as f:
        # This is a list of `imitation.data.types.Trajectory`, where
        # every instance contains observations and actions for a single expert
        # demonstration.
        trajectories = pickle.load(f)
    # Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
    # This is a more general dataclass containing unordered
    # (observation, actions, next_observation) transitions.
    transitions = rollout.flatten_trajectories(trajectories)

    venv = util.make_vec_env(env_name, n_envs=2)

    # tempdir = tempfile.TemporaryDirectory(prefix="il_results/{}_{}".format(rl_algo_name, env_name))
    # tempdir_path = pathlib.Path(tempdir.name)
    # print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")
    log_path = "il_results/{}_{}/{}/".format(rl_algo_name, env_name, imitation_algo_name)

    if imitation_algo_name == 'BC':
        # Train BC on expert data.
        # BC also accepts as `expert_data` any PyTorch-style DataLoader that iterates over
        # dictionaries containing observations and actions.
        logger.configure(log_path)
        trainer = bc.BC(venv.observation_space, venv.action_space, expert_data=transitions)
        trainer.train(n_epochs=3)


    elif imitation_algo_name == 'GAIL':
        logger.configure(log_path)
        gail_trainer = adversarial.GAIL(
            venv,
            expert_data=transitions,
            expert_batch_size=32,
            gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024),
        )
        gail_trainer.train(total_timesteps=2048)
    elif imitation_algo_name == 'AIRL':
        # Train AIRL on expert data.
        logger.configure(log_path)
        airl_trainer = adversarial.AIRL(
            venv,
            expert_data=transitions,
            expert_batch_size=32,
            gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024),
        )
        airl_trainer.train(total_timesteps=2048)

    sample_until = rollout.min_episodes(15)
    trained_ret_mean = rollout.mean_return(trainer.policy, venv, sample_until)
    trainer.save_policy("{}/bc_policy.pth.tar".format(log_path))

    return trained_ret_mean


if __name__=='__main__':
    rl_algo_name = 'ppo'
    env_name = "CartPole-v1"
    expert_traj_path = "expert/{}_{}/rollouts/final.pkl".format(rl_algo_name, env_name)
    imitation_algo = 'BC'
    # imitation_algo = 'GAIL'

    trained_ret_mean = imitation_learning(expert_traj_path, imitation_algo, rl_algo_name, env_name)
    print("trained_ret_mean:{}".format(trained_ret_mean))

