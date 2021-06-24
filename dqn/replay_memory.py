
import numpy as np


class ReplayMemory():
    """
    Replay memory data buffer for storing past transitions
    """
    def __init__(self, n, obs_shape):
        n = int(n)
        self.n = n
        self.o = np.zeros((n, *obs_shape))
        self.a = np.zeros(n)
        self.r = np.zeros(n)
        self.o2 = np.zeros((n, *obs_shape))
        self.d = np.full(n, False)
        self.num_stores = 0

    def store(self, obs, action, rew, obs2, done):
        i = self.num_stores % self.n
        self.o[i] = obs
        self.a[i] = action
        self.r[i] = rew
        self.o2[i] = obs2
        self.d[i] = done
        self.num_stores += 1

    def sample(self, minibatch_size):
        """
        Sample a random minibatch of transitions
        :return:
        """
        active_storage_size = min(self.num_stores, self.n)
        if minibatch_size > active_storage_size:
            raise ValueError(f"Not enough samples in replay memory ({active_storage_size}) to sample minibatch_size "
                             f"({minibatch_size})")

        indices = np.random.choice(range(active_storage_size), minibatch_size, replace=False)
        o = self.o[indices]
        a = self.a[indices]
        r = self.r[indices]
        o2 = self.o2[indices]
        d = self.d[indices]

        return o, a, r, o2, d


def initialize_replay_memory(n_steps, env, replay_memory):
    """
    Initialize with random steps from env

    :param n_steps:
    :param env:
    :param replay_memory:
    :return:
    """
    obs = env.reset()
    for step in range(n_steps):
        a = env.action_space.sample()
        obs2, r, d, _ = env.step(a)

        replay_memory.store(obs, a, r, obs2, d)

        # For next iteration
        obs = env.reset() if d else obs2
