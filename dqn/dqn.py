
import gym
import numpy as np
from copy import deepcopy

import torch as th
import torch.nn as nn
from torchvision import transforms


ATARI_OBS_SHAPE = (210, 160, 3)
OBS_SEQUENCE_LENGTH = 4  # number of frames to keep as "last N frames" to feed as input to Q network
# Need different image cropping (roughly capturing the playing area of screen) for each env; starting row for crop
CROP_START_ROW = {"Pong-v0": 18}


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
        self.q = QNetwork(n_actions)
        self.q_target = deepcopy(self.q)

        # Need different image cropping (roughly capturing the playing area of screen) for each env
        game = env.spec.id
        crop_start_row = CROP_START_ROW[game]
        self.preprocess_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((110, 84)),
            SimpleCrop(crop_start_row, 0, 84, 84)
        ])

        # Instantiate replay memory, first getting p_obs_seq shape
        sampled_obs = self.env.observation_space.sample()
        obs_seq = [sampled_obs] * OBS_SEQUENCE_LENGTH
        p_obs_seq = self._preprocess_obs_sequence(obs_seq)
        self.replay_memory = ReplayMemory(replay_memory_size, p_obs_seq.shape)

    def learn(
        self,
        n_steps,
        epsilon,
        gamma,
        minibatch_size,
        target_update_steps,
        initial_non_update_steps,
        lr=1e-3
    ):
        """
        """
        optimizer_q = th.optim.RMSprop(self.q.parameters(), lr=lr)

        o = self.env.reset()
        # Last 4 frames; duplicates earliest b/c not enough frames in history yet
        latest_obs_sequence = [o] * OBS_SEQUENCE_LENGTH
        pos = self._preprocess_obs_sequence(latest_obs_sequence)
        for step in range(n_steps):
            # Take step and store transition in replay memory
            if np.random.random() > epsilon:
                a = self.predict(pos.unsqueeze(0))
            else:
                a = self.env.action_space.sample()
            o2, r, d, _ = self.env.step(a)

            latest_obs_sequence.pop(0)
            latest_obs_sequence.append(o2)
            pos2 = self._preprocess_obs_sequence(latest_obs_sequence)
            self.replay_memory.store(pos, a, r, pos2, d)

            # For next iteration
            pos = self.env.reset() if d else pos2

            # Use minibatch sampled from replay memory to take grad descent step (after completed initial steps)
            if step >= initial_non_update_steps:
                posm, am, rm, pos2m, dm = self.replay_memory.sample(minibatch_size)  # "m" means "minibatch samples"
                ym = rm + dm * gamma * th.max(self.q_target(pos2m), dim=1)
                # TODO: remove
                assert tuple(ym.shape) == (minibatch_size,)
                assert tuple(am.shape) == (minibatch_size,)
                # Obtain Q values by selecting actions (am) individually for each row of the minibatch
                predm = self.q(posm)[range(minibatch_size), am]
                assert tuple(predm.shape) == (minibatch_size,) # TODO: remove
                loss = self._compute_loss(predm, ym)

                optimizer_q.zero_grad()
                loss.backward()
                optimizer_q.step()

            if step % target_update_steps == 0:
                self.q_target = deepcopy(self.q)

    def predict(self, preprocessed_obs):
        # TODO: might need to specify dim arg
        action = th.argmax(self.q(preprocessed_obs))
        return action

    @staticmethod
    def _compute_loss(predictions, targets):
        loss = (predictions - targets) ** 2
        return loss

    def _preprocess_obs_sequence(self, obs_seq):
        assert len(obs_seq) == OBS_SEQUENCE_LENGTH
        for a in obs_seq:
            assert a.shape == ATARI_OBS_SHAPE

        p_obs_seq = th.tensor(obs_seq).float()
        p_obs_seq = p_obs_seq.permute(0, 3, 1, 2)
        p_obs_seq = self.preprocess_transform(p_obs_seq)
        # Squeeze out grayscale dimension (original RGB dim)
        p_obs_seq = p_obs_seq.squeeze(1)
        # TODO: remove
        assert tuple(p_obs_seq.shape) == (4, 84, 84)
        return p_obs_seq


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


class SimpleCrop(th.nn.Module):
    """
    Crops an image (deterministically) using the TF.crop function. (No simple crop can be found in the
    torchvision.transforms library
    """
    def __init__(self, i, j, h, w):
        super().__init__()
        self.i = i
        self.j = j
        self.h = h
        self.w = w

    def forward(self, img):
        return transforms.functional.crop(img, self.i, self.j, self.h, self.w)


def annealed_epsilon(step, epsilon_start, epsilon_stop, anneal_finished_step):
    return epsilon_start + (epsilon_stop - epsilon_start) * min(1, step / anneal_finished_step)






