
import gym
import torch as th
import torch.nn as nn
import numpy as np


class DQN():
    """
    A working implementation that reproduces DQN for Atari, based entirely from the original paper
    (https://arxiv.org/pdf/1312.5602.pdf).

    """
    def __init__(self, env):
        self.env = env
        self.q = QNetwork()
        self.q_target = QNetwork()
        self.replay_memory = ReplayMemory()

    def learn(self, num_steps, epsilon, gamma, minibatch_size, target_update_steps, lr=1e-3):
        """

        :param steps:
        :return:
        """
        optimizer_q = th.optim.RMSprop(self.q.parameters(), lr=lr)

        po = self._preprocess_obs(self.env.reset())
        for step in range(num_steps):
            if np.random.random() > epsilon:
                a = self.predict(po)
            else:
                a = self.env.action_space.sample()
            o2, r, d, _ = self.env.step(a)
            po2 = self._preprocess_obs(o2)
            self.replay_memory.store(po, a, r, po2, d)

            po = po2  # for next iteration

            # Use minibatch sampled from replay memory to take grad descent step
            pom, am, rm, po2m, dm = self.replay_memory.sample(minibatch_size)  # "m" means "minibatch samples"
            y = rm + dm * gamma * th.max(self.q_target(po2m))  # TODO: ensure batch; also might need specify dim
            pred = self.predict(pom)
            loss = self._compute_loss(pred, y)

            optimizer_q.zero_grad()
            loss.backward()
            optimizer_q.step()

            if step % target_update_steps == 0:
                self.q_target = self.q

    def predict(self, preprocessed_obs):
        # TODO might want to mandata first dim is batch_dim
        # TODO: might need to specify dim arg
        action = th.argmax(self.q(preprocessed_obs))
        return action

    def _compute_loss(self, predictions, targets):
        pass

    def _preprocess_obs(self, obs):
        pass


class ReplayMemory():
    """
    Replay memory data buffer for storing past transitions
    """
    def __init__(self, n):
        pass

    def store(self, p_obs, action, rew, p_obs2):
        pass

    def sample(self, batch_size):
        """
        Sample a random minibatch of transitions
        :return:
        """
        pass


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


def annealed_epsilon(step, epsilon_start, epsilon_stop, anneal_finished_step):
    return epsilon_start + (epsilon_stop - epsilon_start) * min(1, step / anneal_finished_step)






