"""
Benchmarking DQN against simple environments. Run this file as a script (`python benchmark_tests.py`). Will save
diagnostic data continuously to tensorboard (`tb_logs` directory).
"""

from functools import partial
from typing import Any, Dict, Tuple

import gym
import numpy as np
import torch as th
from dqn.dqn import DQN
from dqn.tests import TESTS_DIR
from dqn.utils import annealed_epsilon, datetime_string
from torch import nn


def benchmark_against_cartpole() -> None:
    """
    Benchmark dqn against CartPole. Will save progress continuously to tensorboard logs and will also save each
    final model
    """
    env = gym.make("CartPole-v1")
    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    learn_steps = int(3e6)
    for trial in range(10):
        q_network = nn.Sequential(
            nn.Linear(n_inputs, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_actions)
        )

        dt = datetime_string()

        model = DQN(
            env,
            replay_memory_size=int(500e3),
            q_network=q_network,
            tb_log_dir=f"{TESTS_DIR}/tb_logs/{dt}_benchmark_cartpole_trial{trial}",
        )
        epsilon_fn = partial(
            annealed_epsilon,
            epsilon_start=1,
            epsilon_stop=0.05,
            anneal_finished_step=50e3,
        )
        model.learn(
            learn_steps,
            epsilon=epsilon_fn,
            gamma=0.99,
            batch_size=256,
            update_freq=4,
            target_update_freq=int(2.5e3),
            lr=1e-4,
            initial_replay_memory_steps=int(50e3),
            initial_no_op_actions_max=0,
            eval_freq=int(2.5e3),
            eval_num_episodes=10,
            optimizer_cls=th.optim.Adam,
        )

        th.save(model.q.state_dict(), f"{TESTS_DIR}/models/{dt}_benchmark_cartpole_trial{trial}")


def benchmark_against_frozenlake() -> None:
    """
    Benchmark dqn against FrozenLake. Will save progress continuously to tensorboard logs and will also save each
    final model
    """
    raw_env = gym.make("FrozenLake-v0")
    env = EnvWithObsAsArray(raw_env)

    n_inputs = 1
    n_actions = env.action_space.n

    learn_steps = int(3e6)
    for trial in range(10):
        q_network = nn.Sequential(
            nn.Linear(n_inputs, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_actions)
        )

        dt = datetime_string()

        model = DQN(
            env,
            replay_memory_size=int(500e3),
            q_network=q_network,
            tb_log_dir=f"{TESTS_DIR}/tb_logs/{dt}_benchmark_frozenlake_trial{trial}",
        )
        epsilon_fn = partial(
            annealed_epsilon,
            epsilon_start=1,
            epsilon_stop=0.05,
            anneal_finished_step=50e3,
        )
        model.learn(
            learn_steps,
            epsilon=epsilon_fn,
            gamma=0.99,
            batch_size=256,
            update_freq=4,
            target_update_freq=int(2.5e3),
            lr=1e-4,
            initial_replay_memory_steps=int(50e3),
            initial_no_op_actions_max=0,
            eval_freq=int(2.5e3),
            eval_num_episodes=10,
            optimizer_cls=th.optim.Adam,
        )

        th.save(model.q.state_dict(), f"{TESTS_DIR}/models/{dt}_benchmark_frozenlake_trial{trial}")


class EnvWithObsAsArray(gym.Env):
    """
    Wrapper environment that wraps observations as numpy arrays (required for feeding into the model). E.g. useful
    for wrapping FrozenLake env
    """

    def __init__(self, env: gym.Env):
        super().__init__()
        self.env = env
        assert isinstance(self.env.observation_space, gym.spaces.Discrete)
        self.observation_space = gym.spaces.Box(0, self.env.observation_space.n, (1,), dtype=int)
        self.action_space = self.env.action_space

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, rew, done, info = self.env.step(action)
        assert isinstance(obs, (int, float))
        obs_arr = np.array([obs])
        return obs_arr, rew, done, info

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        assert isinstance(obs, (int, float))
        obs_arr = np.array([obs])
        return obs_arr

    def render(self, mode="human") -> None:
        self.env.render(mode)


def main():
    benchmark_against_cartpole()
    benchmark_against_frozenlake()


if __name__ == "__main__":
    main()
