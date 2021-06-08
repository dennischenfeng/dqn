
import gym
import numpy as np
from copy import deepcopy

import torch as th
import torch.nn as nn
from torchvision import transforms


ATARI_OBS_SHAPE = (210, 160, 3)
OBS_MAXED_SEQUENCE_LENGTH = 4  # number of obs_maxed's to keep as "last N frames" to feed as input to Q network
# Need different image cropping (roughly capturing the playing area of screen) for each env; starting row for crop
CROP_START_ROW = {"PongNoFrameskip-v4": 18}


class DQN():
    """
    A working implementation that reproduces DQN for Atari, based entirely from the original Nature paper

    """
    def __init__(self, env, replay_memory_size=1e6, action_repeat=4):
        if not isinstance(env.action_space, gym.spaces.discrete.Discrete):
            raise ValueError("`Environment action space must be `Discrete`; DQN does not support otherwise.")
        self.env = env
        self.action_repeat = action_repeat
        n_actions = self.env.action_space.n
        self.q = QNetwork(n_actions)
        self.q_target = deepcopy(self.q)

        # Need different image cropping (roughly capturing the playing area of screen) for each env
        game = env.spec.id
        crop_start_row = CROP_START_ROW[game]
        self.preprocess_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((110, 84)),
            SimpleCrop(crop_start_row, 0, 84, 84)  # TODO: confirm that Nature paper crops differently for each game
        ])

        # Instance variables for tracking state while stepping through env
        self.prev_obs = None
        self.latest_obs_maxed_seq = []

        # Instantiate replay memory with mod_obs shape
        mod_obs = self.mod_env_reset()
        self.replay_memory = ReplayMemory(replay_memory_size, mod_obs.shape)

    def learn(
        self,
        n_steps,
        epsilon,
        gamma,
        batch_size,
        target_update_steps,
        initial_non_update_steps,
        lr=1e-3
    ):
        """
        """
        optimizer_q = th.optim.RMSprop(self.q.parameters(), lr=lr)

        mod_obs = self.mod_env_reset()
        for step in range(n_steps):
            # Take step and store transition in replay memory
            if np.random.random() > epsilon:
                a = self.predict(mod_obs.unsqueeze(0)).item()
            else:
                a = self.env.action_space.sample()
            mod_obs2, r, d, _ = self.mod_env_step(a)

            self.replay_memory.store(mod_obs, a, r, mod_obs2, d)

            # For next iteration
            mod_obs = self.mod_env_reset() if d else mod_obs2

            # Use minibatch sampled from replay memory to take grad descent step (after completed initial steps)
            if step >= initial_non_update_steps:
                mod_obsb, ab, rb, mod_obs2b, db = self.replay_memory.sample(batch_size)  # `b` means "batch"
                mod_obsb, rb, mod_obs2b, db = list(map(lambda x: th.tensor(x).float(), [mod_obsb, rb, mod_obs2b, db]))
                ab = th.tensor(ab).long()

                yb = rb + db * th.tensor(gamma) * th.max(self.q_target(mod_obs2b), dim=1).values
                # TODO: remove
                assert tuple(yb.shape) == (batch_size,)
                assert tuple(ab.shape) == (batch_size,)
                # Obtain Q values by selecting actions (am) individually for each row of the minibatch
                predb = self.q(mod_obsb)[th.arange(batch_size), ab]
                assert tuple(predb.shape) == (batch_size,)  # TODO: remove
                loss = self.compute_loss(predb, yb)

                optimizer_q.zero_grad()
                loss.backward()
                optimizer_q.step()

            if step % target_update_steps == 0:
                self.q_target = deepcopy(self.q)

    def mod_env_reset(self):
        """

        :return:
        """
        obs = self.env.reset()
        # For first obs, maxing over current & "previous" is meaningless; use obs itself
        obs_maxed = obs
        self.latest_obs_maxed_seq = [obs_maxed] * OBS_MAXED_SEQUENCE_LENGTH

        # For next iteration
        self.prev_obs = obs

        mod_obs = self.preprocess_obs_maxed_seq()
        return mod_obs

    def mod_env_step(self, action):
        """

        :return:
        """

        obs_maxed = None
        total_rew = 0
        done = None
        info = None
        assert self.action_repeat >= 1
        for i in range(self.action_repeat):
            obs2, rew, done, info = self.env.step(action)

            obs_maxed = np.maximum(self.prev_obs, obs2)
            # As discussed in the paper, clip step rewards at -1 and +1 to limit scale of errors (potentially better
            # training stability), but reduces ability to differentiate actions for large/small rewards
            total_rew += float(np.clip(rew, -1, 1))
            if done:
                break

            # For next iteration
            self.prev_obs = obs2

        self.latest_obs_maxed_seq.pop(0)
        self.latest_obs_maxed_seq.append(obs_maxed)
        mod_obs = self.preprocess_obs_maxed_seq()

        return mod_obs, total_rew, done, info

    def predict(self, p_obs_seq_batched):
        action = th.argmax(self.q(p_obs_seq_batched), dim=1)
        return action

    @staticmethod
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

    def preprocess_obs_maxed_seq(self):
        assert len(self.latest_obs_maxed_seq) == OBS_MAXED_SEQUENCE_LENGTH
        for a in self.latest_obs_maxed_seq:
            assert a.shape == ATARI_OBS_SHAPE

        result = th.tensor(self.latest_obs_maxed_seq).float()
        result = result.permute(0, 3, 1, 2)
        result = self.preprocess_transform(result)
        # Squeeze out grayscale dimension (original RGB dim)
        result = result.squeeze(1)
        # TODO: remove
        assert tuple(result.shape) == (4, 84, 84)
        return result


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
            nn.Linear(n_flattened_activations, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.net(x)


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






