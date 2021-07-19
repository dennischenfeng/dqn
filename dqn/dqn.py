import gym
import numpy as np
from copy import deepcopy
from typing import Optional, Dict
import torch as th
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dqn.replay_memory import ReplayMemory, initialize_replay_memory
from dqn.preprocessed_atari_env import OBS_MAXED_SEQUENCE_LENGTH
from dqn.utils import evaluate_model, annealed_epsilon
from dqn.callbacks import BaseCallback
from typing import Optional, Union, Callable, Any
from torch.nn import functional as F
from functools import partial

NATURE_Q_NETWORK_ALLOWED_CHANNELS = (1, 3, 4)


class DQN:
    """
    A working implementation that reproduces DQN for Atari, based entirely from the original Nature paper
    """

    def __init__(
        self,
        env: gym.Env,
        q_network: Optional[nn.Module] = None,
        replay_memory_size: int = 1e6,
        tb_log_dir: Optional[str] = None,
        device: Optional[th.device] = None,
    ):
        """
        :param env: environment
        :param q_network: Q network (action-value). Num inputs must be observation space shape, num outputs must be
            action_space shape
        :param replay_memory_size: total number of transitions that can be stored in replay memory
        :param tb_log_dir: tensorboard log directory path. If None, won't log diagnostics.
        :param device: device (cpu, cuda, etc) to place tensors on
        """
        if not isinstance(env.action_space, gym.spaces.discrete.Discrete):
            raise ValueError("`Environment action space must be `Discrete`; DQN does not support otherwise.")
        self.env = env
        self.eval_env = deepcopy(env)

        # device
        if device is None:
            self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # q_network
        if q_network is None:
            self.q = NatureQNetwork(env.observation_space, env.action_space).to(self.device)
        else:
            self.q = q_network.to(self.device)
        self.q_target = deepcopy(self.q).to(self.device)

        # Instantiate replay memory with mod_obs shape
        mod_obs = self.env.reset()
        self.replay_memory = ReplayMemory(replay_memory_size, mod_obs.shape, mod_obs.dtype)

        # Tensorboard writer
        self.tb_log_dir = tb_log_dir
        self.writer = None
        if self.tb_log_dir:
            self.writer = SummaryWriter(tb_log_dir, flush_secs=30)

    def learn(
        self,
        n_steps: int,
        lr: float = 1e-3,
        epsilon: Optional[Union[int, float, Callable[[int], float]]] = None,
        gamma: float = 0.99,
        batch_size: int = 32,
        update_freq: int = 4,
        target_update_freq: int = 2500,
        initial_replay_memory_steps: int = 50000,
        optimizer_cls: th.optim.Optimizer = th.optim.RMSprop,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        initial_no_op_actions_max: int = 0,
        eval_freq: int = 1000,
        eval_num_episodes: int = 10,
        callback: Optional[BaseCallback] = None,
    ) -> None:
        """
        :param n_steps: num env steps
        :param lr: learning rate
        :param epsilon: float (constant epsilon) or function (epsilon is a function of num steps taken),
            for epsilon-greedy algorithm for choosing action. If None, will choose default annealed epsilon from 1 to
            0.1 with final annealed step of 50000.
        :param gamma: discount factor
        :param batch_size: minibatch size for network updates
        :param update_freq: update the q network every `update_freq` env steps
        :param target_update_freq: update the target network every `target_update_freq` q updates (not env steps)
        :param initial_replay_memory_steps: num random env transitions to store into replay memory, before starting
            any network updates. These initial step do not count toward the "step" count
        :param optimizer_cls: optimizer class
        :param optimizer_kwargs: kwargs for optimizer class. If None, will choose empty kwargs dict
        :param initial_no_op_actions_max: each episode, the first n actions will be no-operation; n is randomly
            between 0 and this number
        :param eval_freq: run an evaluation (i.e. test episodes) every `eval_freq` q updates. Only run if using
            tensorboard logging
        :param eval_num_episodes: num episodes for evaluation. Only run if using tensorboard logging
        :param callback: callback object for running callbacks during training
        """
        epsilon_fn = construct_epsilon_fn(epsilon)
        initialize_replay_memory(initial_replay_memory_steps, self.env, self.replay_memory)
        if optimizer_kwargs is None:
            optimizer_kwargs = dict()

        initial_no_op_actions = np.random.randint(initial_no_op_actions_max + 1)
        optimizer_q = optimizer_cls(self.q.parameters(), lr=lr, **optimizer_kwargs)

        num_updates = 0
        ep_rew = 0
        ep_length = 0
        obs = self.env.reset()

        for step in range(n_steps):
            # Take step and store transition in replay memory
            # TODO: continue here: initial no op actions needs to be reset every episode
            if step < initial_no_op_actions:
                a = 0
            elif np.random.random() > epsilon_fn(step):
                a = self.predict(obs)
            else:
                a = self.env.action_space.sample()
            obs2, r, d, _ = self.env.step(a)
            ep_rew += r
            ep_length += 1

            self.replay_memory.store(obs, a, r, obs2, d)

            # For next iteration
            if d:
                obs = self.env.reset()
                if self.tb_log_dir:
                    self.writer.add_scalar("train_ep_rew", ep_rew, step)
                    self.writer.add_scalar("train_ep_length", ep_length, step)
                ep_rew = 0
                ep_length = 0
            else:
                obs = obs2

            # Use minibatch sampled from replay memory to take grad descent step (after completed initial steps)
            if step % update_freq == 0:
                obsb, ab, rb, obs2b, db = self.replay_memory.sample(batch_size)  # `b` means "batch"
                obsb, rb, obs2b, db = list(map(lambda x: th.tensor(x).float().to(self.device), [obsb, rb, obs2b, db]))
                ab = th.tensor(ab).long().to(self.device)

                # Avoid gradient calculations through q_target
                with th.no_grad():
                    yb = rb + (1 - db) * gamma * th.max(self.q_target(obs2b), dim=1).values
                # Select actions (ab) individually for each minibatch item. Compared to using th.gather to select
                # actions, this indexing takes about the same computational time for backward propagation
                predb = self.q(obsb)[th.arange(batch_size), ab]
                # Use smooth L1 loss, as discussed in paper
                loss = F.smooth_l1_loss(predb, yb)

                optimizer_q.zero_grad()
                loss.backward()
                # TODO: test grad norm clipping
                th.nn.utils.clip_grad_norm_(self.q.parameters(), 10)
                optimizer_q.step()
                num_updates += 1

                if num_updates % target_update_freq == 0:
                    self.q_target = deepcopy(self.q).to(self.device)

                if num_updates % eval_freq == 0 and self.tb_log_dir:
                    self.writer.add_scalar("train_q_mean", predb.mean().item(), step)
                    self.writer.add_scalar("train_loss", loss.item(), step)
                    self.writer.add_scalar("epsilon", epsilon_fn(step), step)

                    ep_rews = evaluate_model(self, self.eval_env, num_episodes=eval_num_episodes)
                    self.writer.add_scalar("eval_ep_rew_mean", np.mean(ep_rews), step)
                    self.writer.add_scalar("eval_ep_rew_max", np.max(ep_rews), step)
                    self.writer.add_scalar("eval_ep_rew_min", np.min(ep_rews), step)
                    self.writer.add_scalar("eval_ep_rew_std", np.std(ep_rews), step)

            # Callback
            if callback:
                callback.after_step(locals(), globals())

    def predict(self, obs: Any) -> Any:
        """
        :param obs: observation
        :return:
        """
        with th.no_grad():
            obs_t = th.tensor(obs).float().to(self.device)
            obs_t = obs_t.unsqueeze(0)
            action = th.argmax(self.q(obs_t), dim=1)
            action = action.item()
        return action


def construct_epsilon_fn(epsilon: Optional[Union[int, float, Callable[[int], float]]]) -> Callable[[int], float]:
    """
    Helper function for constructing epsilon function for DQN's learn method

    :param epsilon: (see learn method for details)
    :return: epsilon function
    """
    if epsilon is None:
        epsilon_fn = partial(
            annealed_epsilon, epsilon_start=1.0, epsilon_stop=0.1,
            anneal_finished_step=50000
        )
    elif isinstance(epsilon, (float, int)):
        def epsilon_fn(_):
            return float(epsilon)
    else:
        if callable(epsilon):
            epsilon_fn = epsilon
        else:
            raise ValueError(
                "`epsilon` must be either a float/int or a callable that returns a float/int when given the "
                "step_index.")

    return epsilon_fn


class NatureQNetwork(nn.Module):
    """
    The CNN Q network described in the paper
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        """
        requires image obs are ordered like "CxHxW"

        :param observation_space: obs space of env
        :param action_space: action space of env
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

    # TODO: continue with type hinting here
    def forward(self, obs):
        obs_normed = normalize_image_obs(obs)
        return self.fc(self.cnn(obs_normed))


def normalize_image_obs(obs):
    return obs / 255.0

