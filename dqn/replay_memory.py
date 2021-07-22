"""
Replay memory used to store transition data.
"""

import numpy as np
from typing import Tuple
import gym


class ReplayMemory:
    """
    Replay memory data buffer for storing past transitions
    """

    def __init__(self, n: int, obs_shape: Tuple[int, ...], obs_dtype: np.dtype) -> None:
        """
        :param n: storage capacity of replay memory (number of transitions)
        :param obs_shape: shape of a signal transition's observation array
        :param obs_dtype: datatype of elements in observation array
        """
        n = int(n)
        self.n = n
        self.o = np.zeros((n, *obs_shape), dtype=obs_dtype)
        self.a = np.zeros(n)
        self.r = np.zeros(n)
        self.o2 = np.zeros((n, *obs_shape), dtype=obs_dtype)
        self.d = np.full(n, False)
        self.num_stores = 0

    def store(self, obs: np.ndarray, action: int, rew: float, obs2: np.ndarray, done: bool) -> None:
        """
        Store a transition.

        :param obs: observation
        :param action: action
        :param rew: reward
        :param obs2: next observation
        :param done: whether done
        """
        i = self.num_stores % self.n
        self.o[i] = obs
        self.a[i] = action
        self.r[i] = rew
        self.o2[i] = obs2
        self.d[i] = done
        self.num_stores += 1

    def sample(self, batch_size) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        """
        Sample a random minibatch of transitions (sampling with replacement)

        :param batch_size: number of items in minibatch
        """
        active_storage_size = min(self.num_stores, self.n)
        if batch_size > active_storage_size:
            raise ValueError(f"Not enough samples in replay memory ({active_storage_size}) to sample minibatch_size "
                             f"({batch_size})")

        indices = np.random.randint(0, active_storage_size, size=batch_size)
        o = self.o[indices]
        a = self.a[indices]
        r = self.r[indices]
        o2 = self.o2[indices]
        d = self.d[indices]

        return o, a, r, o2, d


def initialize_replay_memory(n_steps: int, env: gym.Env, replay_memory: ReplayMemory) -> None:
    """
    Initialize with random steps from env

    :param n_steps: number of steps
    :param env: gym environment
    :param replay_memory: ReplayMemory object to initialize (will mutate)
    """
    obs = env.reset()
    for _ in range(n_steps):
        a = env.action_space.sample()
        obs2, r, d, _ = env.step(a)

        replay_memory.store(obs, a, r, obs2, d)

        # For next iteration
        obs = env.reset() if d else obs2
