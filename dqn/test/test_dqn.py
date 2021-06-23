import pytest
import numpy as np
import torch as th
import torch.nn as nn
import gym
import mock
from functools import partial
from dqn.dqn import DQN, NatureQNetwork, compute_loss
from dqn.preprocessed_atari_env import PreprocessedAtariEnv, OBS_MAXED_SEQUENCE_LENGTH, MOD_OBS_SHAPE, \
    ReorderedObsAtariEnv
from dqn.utils import evaluate_model, annealed_epsilon, basic_mlp_network


def test_nature_q_network():
    for image_size in [[40, 40], [50, 50], [40, 50]]:
        for n_input_channels in [1, 3, 4]:
            obs_space = gym.spaces.Box(0, 255, shape=(n_input_channels, *image_size), dtype=np.uint8)

            for n_actions in [5, 10, 15]:
                action_space = gym.spaces.Discrete(n_actions)

                for minibatch_size in [16, 32, 64]:
                    q = NatureQNetwork(obs_space, action_space)
                    mod_obs_batch = th.ones((minibatch_size, *obs_space.shape)).float()
                    assert q(mod_obs_batch).shape == (minibatch_size, n_actions)


def test_dqn_compute_loss():
    targets = th.tensor([
        [1.0],
        [2.0],
        [3.0]
    ])

    preds = th.tensor([
        [1.1],
        [3.0],
        [13.0]
    ])

    assert compute_loss(preds, targets) == pytest.approx(3.67)


def mock_preprocess_obs_maxed_seq(self):
    result = th.tensor(self.latest_obs_maxed_seq)
    return result


def test_dqn():
    raw_env = gym.make("PongNoFrameskip-v4")
    env = PreprocessedAtariEnv(raw_env)
    model = DQN(env, replay_memory_size=100)

    model.learn(
        34, epsilon=0.1, gamma=0.99, batch_size=32, update_freq=4, target_update_freq=10, lr=1e-3,
        initial_non_update_steps=32, initial_no_op_actions_max=30
    )


def test_dqn_annealed_epsilon():
    raw_env = gym.make("PongNoFrameskip-v4")
    env = PreprocessedAtariEnv(raw_env)
    model = DQN(env, replay_memory_size=100)

    steps = 34
    epsilon_fn = partial(annealed_epsilon, epsilon_start=1, epsilon_stop=0.1, anneal_finished_step=30)

    model.learn(
        steps, epsilon=epsilon_fn, gamma=0.99, batch_size=32, update_freq=4, target_update_freq=10, lr=1e-3,
        initial_non_update_steps=32, initial_no_op_actions_max=30
    )


def test_dqn_orig_pong_env():
    raw_env = gym.make("PongNoFrameskip-v4")
    env = ReorderedObsAtariEnv(raw_env)

    model = DQN(env, replay_memory_size=100)

    model.learn(
        34, epsilon=0.1, gamma=0.99, batch_size=32, update_freq=4, target_update_freq=10, lr=1e-3,
        initial_non_update_steps=32, initial_no_op_actions_max=30
    )


def test_dqn_cartpole_env():
    env = gym.make("CartPole-v1")
    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    q_network = basic_mlp_network(n_inputs, n_actions)

    model = DQN(env, q_network=q_network, replay_memory_size=100)

    model.learn(
        34, epsilon=0.1, gamma=0.99, batch_size=32, update_freq=4, target_update_freq=10, lr=1e-3,
        initial_non_update_steps=32, initial_no_op_actions_max=30
    )


def test_dqn_cartpole_env_tb(tmpdir):
    tmpdir = str(tmpdir)

    env = gym.make("CartPole-v1")
    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    q_network = basic_mlp_network(n_inputs, n_actions)

    model = DQN(env, q_network=q_network, replay_memory_size=100, tb_log_dir=tmpdir)

    model.learn(
        34, epsilon=0.1, gamma=0.99, batch_size=32, update_freq=4, target_update_freq=10, lr=1e-3,
        initial_non_update_steps=32, initial_no_op_actions_max=30
    )

