import gym
import numpy as np
from copy import deepcopy
import torch as th
import torch.nn as nn
from dqn.replay_memory import ReplayMemory
from dqn.preprocessed_atari_env import OBS_MAXED_SEQUENCE_LENGTH

NATURE_Q_NETWORK_ALLOWED_CHANNELS = (1, 3, 4)


class DQN():
    """
    A working implementation that reproduces DQN for Atari, based entirely from the original Nature paper

    """
    def __init__(self, env, q_network=None, replay_memory_size=1e6):
        if not isinstance(env.action_space, gym.spaces.discrete.Discrete):
            raise ValueError("`Environment action space must be `Discrete`; DQN does not support otherwise.")
        self.env = env
        if q_network is None:
            self.q = NatureQNetwork(env.observation_space, env.action_space)
        else:
            self.q = q_network
        self.q_target = deepcopy(self.q)

        # Instantiate replay memory with mod_obs shape
        mod_obs = self.env.reset()
        self.replay_memory = ReplayMemory(replay_memory_size, mod_obs.shape)

    def learn(
        self,
        n_steps,
        epsilon,
        gamma,
        batch_size,
        update_freq,
        target_update_freq,
        initial_non_update_steps,
        initial_no_op_actions_max,
        optimizer_cls=th.optim.RMSprop,
        lr=1e-3
    ):
        """

        :param n_steps:
        :param epsilon:
        :param gamma:
        :param batch_size:
        :param update_freq: in units of steps
        :param target_update_freq: in units of q updates
        :param initial_non_update_steps:
        :param optimizer_cls:
        :param lr:
        :return:
        """
        optimizer_q = optimizer_cls(self.q.parameters(), lr=lr)

        # TODO: implement no op actions at beginning of game

        obs = self.env.reset()
        for step in range(n_steps):
            # Take step and store transition in replay memory
            if np.random.random() > epsilon:
                obs_t = th.tensor(obs).float()
                a = self.predict(obs_t.unsqueeze(0)).item()
            else:
                a = self.env.action_space.sample()
            obs2, r, d, _ = self.env.step(a)

            self.replay_memory.store(obs, a, r, obs2, d)

            # For next iteration
            obs = self.env.reset() if d else obs2

            # Use minibatch sampled from replay memory to take grad descent step (after completed initial steps)
            num_updates = 0
            if step % update_freq == 0 and step >= initial_non_update_steps:
                obsb, ab, rb, obs2b, db = self.replay_memory.sample(batch_size)  # `b` means "batch"
                obsb, rb, obs2b, db = list(map(lambda x: th.tensor(x).float(), [obsb, rb, obs2b, db]))
                ab = th.tensor(ab).long()

                yb = rb + db * th.tensor(gamma) * th.max(self.q_target(obs2b), dim=1).values
                # Obtain Q values by selecting actions (am) individually for each row of the minibatch
                predb = self.q(obsb)[th.arange(batch_size), ab]
                loss = compute_loss(predb, yb)

                optimizer_q.zero_grad()
                loss.backward()
                optimizer_q.step()
                num_updates += 1

                if num_updates % target_update_freq == 0:
                    self.q_target = deepcopy(self.q)

    def predict(self, obs):
        """
        requires obs to be in batched form

        :param obs:
        :return:
        """
        action = th.argmax(self.q(obs), dim=1)
        return action


class NatureQNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        """
        requires image obs are ordered like "CxHxW"

        :param observation_space:
        :param action_space:
        """
        super().__init__()
        assert len(observation_space.shape) == 3
        n_input_channels = observation_space.shape[0]
        assert n_input_channels in NATURE_Q_NETWORK_ALLOWED_CHANNELS
        assert isinstance(action_space, gym.spaces.discrete.Discrete)
        n_actions = action_space.n

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        sample_obs = th.tensor(observation_space.sample()).float()
        with th.no_grad():
            n_flattened_activations = self.cnn(sample_obs.unsqueeze(0)).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(n_flattened_activations, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.fc(self.cnn(x))


def compute_loss(predictions, targets):
    """
    Loss function for optimizing Q. As discussed in the paper, clip the squared error's derivative at -1 and +1,
    i.e. loss = error^2 if |error| < 1 else |error|

    :param predictions:
    :param targets:
    :return:
    """
    error = th.abs(predictions - targets)
    loss = th.mean(th.where(error < 1, error ** 2, error))
    return loss





