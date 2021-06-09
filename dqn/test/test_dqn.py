import pytest
import numpy as np
import torch as th
import gym
import mock

from dqn.dqn import DQN, QNetwork, ReplayMemory, ATARI_OBS_SHAPE
from dqn.replay_memory import ReplayMemory
from dqn.utils import SimpleCrop, annealed_epsilon


def test_q_network(mod_obs_batch):
    for n_actions in [5, 10, 15]:
        for minibatch_size in [32, 64, 128]:
            q = QNetwork(n_actions)
            p_obs_seq = mod_obs_batch
            p_obs_seq = th.tensor(p_obs_seq).float()
            assert q(p_obs_seq).shape == (minibatch_size, n_actions)


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

    assert DQN.compute_loss(preds, targets) == pytest.approx(3.67)


def mock_preprocess_obs_maxed_seq(self):
    result = th.tensor(self.latest_obs_maxed_seq)
    return result


# @mock.patch.object(DQN, "preprocess_obs_maxed_seq", mock_preprocess_obs_maxed_seq)
# def test_dqn_mod_env_step():
#     mock_env = mock.MagicMock(spec=gym.Env)
#
#     def mock_step_values():
#         i = 1
#         while True:
#             obs2 = i * np.ones(ATARI_OBS_SHAPE)
#             obs2[0, 0, 0] = 255 - i
#             rew = i * 0.11
#             done = (i % 10 == 0)
#             info = {"a": 0}
#             yield obs2, rew, done, info
#             i += 1
#
#     mock_env.step.side_effect = mock_step_values()
#     mock_env.reset.return_value = np.zeros(ATARI_OBS_SHAPE)
#     mock_env.action_space = gym.spaces.discrete.Discrete(1)
#     mock_env.spec.id = "PongNoFrameskip-v4"
#
#     model = DQN(mock_env, replay_memory_size=100)
#     mod_obs = model.mod_env_reset()
#     assert (np.array(mod_obs) == np.zeros((4, 84, 84))).all()
#
#     mod_obs, rew, done, info = model.mod_env_step(1)
#     assert mock_env.step.call_count == 4
#     # TODO: how do I elegantly test mod_obs is correct after env steps??? --> just continue using
#     #  mock_preprocess_obs_maxed_seq and mention that it has a different shape from true method because I need to
#     #  check that it's correct
#     assert rew == pytest.approx(0.11 + 0.22 + 0.33 + 0.44)
#     assert not done
#     assert info == {"a": 0}
#     pass

    # model.mod_env_step()


def test_dqn():
    env = gym.make("PongNoFrameskip-v4")
    model = DQN(env, replay_memory_size=1e6)

    model.learn(
        34, epsilon=0.1, gamma=0.99, batch_size=32, target_update_steps=10, lr=1e-3, initial_non_update_steps=32
    )



