import pytest
import numpy as np
import torch as th
import gym
import mock

from dqn.dqn import DQN, QNetwork, ReplayMemory, ATARI_OBS_SHAPE
from dqn.replay_memory import ReplayMemory
from dqn.utils import SimpleCrop, annealed_epsilon


def create_p_obs_seq(minibatch_size):
    # minibatch size 32, 4 channels, 84 x 84 image
    return np.ones((minibatch_size, 4, 84, 84))


def test_annealed_epsilon():
    assert annealed_epsilon(0, 1, 0.1, 50) == pytest.approx(1)
    assert annealed_epsilon(25, 1, 0.1, 50) == pytest.approx(0.55)
    assert annealed_epsilon(50, 1, 0.1, 50) == pytest.approx(0.1)
    assert annealed_epsilon(100, 1, 0.1, 50) == pytest.approx(0.1)

    assert annealed_epsilon(100, 1, 0.1, 200) == pytest.approx(0.55)
    assert annealed_epsilon(100, 1, 0, 200) == pytest.approx(0.5)
    assert annealed_epsilon(100, 0.5, 0, 200) == pytest.approx(0.25)


def test_q_network():
    for n_actions in [5, 10, 15]:
        for minibatch_size in [32, 64, 128]:
            q = QNetwork(n_actions)
            p_obs_seq = create_p_obs_seq(minibatch_size)
            p_obs_seq = th.tensor(p_obs_seq).float()
            assert q(p_obs_seq).shape == (minibatch_size, n_actions)


def test_replay_memory_simple_obs():
    def sample_generator():
        i = 1
        while True:
            obs = i * np.ones(5)
            action = i
            rew = i * 100
            obs2 = (i + 1) * np.ones(5)
            done = False
            yield obs, action, rew, obs2, done
            i += 1

    g = sample_generator()

    o, a, r, o2, d = next(g)
    m = ReplayMemory(10, o.shape)

    # No samples stored yet
    with pytest.raises(ValueError):
        m.sample(1)

    # Store 1 sample
    m.store(o, a, r, o2, d)

    o_s, a_s, r_s, o2_s, d_s = m.sample(1)
    assert (o_s[0] == np.ones(5)).all()
    assert a_s[0] == 1
    assert r_s[0] == 100
    assert (o2_s[0] == 2 * np.ones(5)).all()
    assert not d_s[0]

    with pytest.raises(ValueError):
        m.sample(2)

    # Store 10 total (9 more)
    for _ in range(9):
        o, a, r, o2, d = next(g)
        m.store(o, a, r, o2, d)

    o_s, a_s, r_s, o2_s, d_s = m.sample(10)
    assert 1 in a_s  # Sample 1 is still here
    assert 100 in r_s

    with pytest.raises(ValueError):
        m.sample(11)

    # Store 1 more (should overwrite sample 1)
    o, a, r, o2, d = next(g)
    m.store(o, a, r, o2, d)

    o_s, a_s, r_s, o2_s, d_s = m.sample(10)
    assert 1 not in a_s  # sample 1 not here anymore
    assert 100 not in r_s

    with pytest.raises(ValueError):
        m.sample(11)


def test_replay_memory_representative_obs():
    p_obs_seq = create_p_obs_seq(1)[0]
    action = 2
    rew = 50
    p_obs2 = 2 * p_obs_seq
    done = False

    m = ReplayMemory(10, p_obs_seq.shape)
    m.store(p_obs_seq, action, rew, p_obs2, done)

    pos_s, a_s, r_s, pos2_s, d_s = m.sample(1)
    assert (pos_s[0] == p_obs_seq).all()
    assert a_s[0] == action
    assert r_s[0] == rew
    assert (pos2_s[0] == p_obs2).all()
    assert d_s[0] == done


def test_simple_crop():
    img = th.arange(10).float()
    img = img.repeat(20, 1)
    img = img.unsqueeze(0)

    c = SimpleCrop(1, 2, 3, 4)
    cropped_img = c(img)

    expected = th.tensor([[
        [2, 3, 4, 5],
        [2, 3, 4, 5],
        [2, 3, 4, 5]
    ]])
    expected = expected.float()

    assert (expected == cropped_img).all().item()


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



