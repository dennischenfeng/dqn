import pytest
import numpy as np
import torch as th
import gym
from dqn.dqn import DQN, QNetwork, ReplayMemory, annealed_epsilon, SimpleCrop


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


def test_dqn():
    env = gym.make("Pong-v0")
    model = DQN(env, replay_memory_size=1e6)

    model.learn(1, epsilon=0.1, gamma=0.99, minibatch_size=32, target_update_steps=10, lr=1e-3)

