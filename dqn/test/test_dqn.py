import pytest
import numpy as np
import torch as th
import gym
import mock
from dqn.dqn import DQN, NatureQNetwork, compute_loss
from dqn.preprocessed_atari_env import PreprocessedAtariEnv, OBS_MAXED_SEQUENCE_LENGTH, MOD_OBS_SHAPE


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
        34, epsilon=0.1, gamma=0.99, batch_size=32, target_update_steps=10, lr=1e-3, initial_non_update_steps=32
    )


# def test_dqn_orig_pong_env():
#     raw_env = gym.make("PongNoFrameskip-v4")
#
#     model = DQN(raw_env, replay_memory_size=100)
#
#     model.learn(
#         34, epsilon=0.1, gamma=0.99, batch_size=32, target_update_steps=10, lr=1e-3, initial_non_update_steps=32
#     )

# TODO: test with cartpole too