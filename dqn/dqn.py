
import gym
import torch as th
import torch.nn as nn
import numpy as np
from copy import deepcopy

ATARI_OBS_SHAPE = (210, 160, 3)
OBS_SEQUENCE_LENGTH = 4  # number of frames to keep as "last N frames" to feed as input to Q network


class DQN():
    """
    A working implementation that reproduces DQN for Atari, based entirely from the original paper
    (https://arxiv.org/pdf/1312.5602.pdf).

    """
    def __init__(self, env, replay_memory_size):
        if not isinstance(env.action_space, gym.spaces.discrete.Discrete):
            raise ValueError("`Environment action space must be `Discrete`; DQN does not support otherwise.")
        self.env = env
        n_actions = self.env.action_space.n

        # Get p_obs_seq
        sampled_obs = self.env.observation_space.sample()

        self.q = QNetwork(n_actions)
        self.q_target = deepcopy(self.q)
        self.replay_memory = ReplayMemory(replay_memory_size)

    def learn(
        self,
        n_steps,
        epsilon,
        gamma,
        minibatch_size,
        target_update_steps,
        lr=1e-3
    ):
        """
        """
        optimizer_q = th.optim.RMSprop(self.q.parameters(), lr=lr)

        o = self.env.reset()
        # Last 4 frames; duplicates earliest b/c not enough frames in history yet
        latest_obs_sequence = [o] * OBS_SEQUENCE_LENGTH

        pos = self._preprocess_obs_sequence(latest_obs_sequence)
        # TODO: note: ensure data is tensor before feeding into any pytorch function/module
        for step in range(n_steps):
            if np.random.random() > epsilon:
                a = self.predict(pos)
            else:
                a = self.env.action_space.sample()
            o2, r, d, _ = self.env.step(a)
            pos2 = self._preprocess_obs_sequence(o2)
            self.replay_memory.store(pos, a, r, pos2, d)

            # for next iteration
            pos = self.env.reset() if d else pos2

            # Use minibatch sampled from replay memory to take grad descent step
            posm, am, rm, pos2m, dm = self.replay_memory.sample(minibatch_size)  # "m" means "minibatch samples"
            y = rm + dm * gamma * th.max(self.q_target(pos2m))  # TODO: ensure batch; also might need specify dim
            # TODO: need to use `am` to select actions in q
            pred = self.q(posm)
            loss = self._compute_loss(pred, y)

            optimizer_q.zero_grad()
            loss.backward()
            optimizer_q.step()

            if step % target_update_steps == 0:
                self.q_target = deepcopy(self.q)

    def predict(self, preprocessed_obs):
        # TODO: might need to specify dim arg
        action = th.argmax(self.q(preprocessed_obs))
        return action

    def _compute_loss(self, predictions, targets):
        loss = (predictions - targets) ** 2
        return loss

    def _preprocess_obs_sequence(self, obs):
        # TODO: convert to channels first, i.e. CxHxW images
        assert obs.shape == ATARI_OBS_SHAPE


class ReplayMemory():
    """
    Replay memory data buffer for storing past transitions
    """
    def __init__(self, n, obs_shape):
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


class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        n_input_channels = 4
        n_flattened_activations = 3136
        self.net = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_flattened_activations, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, preprocessed_obs):
        return self.net(preprocessed_obs)


def annealed_epsilon(step, epsilon_start, epsilon_stop, anneal_finished_step):
    return epsilon_start + (epsilon_stop - epsilon_start) * min(1, step / anneal_finished_step)






