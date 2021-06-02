import pytest
import numpy as np
import torch as th
from dqn.dqn import QNetwork, ReplayMemory, annealed_epsilon


def create_p_obs(minibatch_size):
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
            p_obs = create_p_obs(minibatch_size)
            p_obs = th.tensor(p_obs).float()
            assert q(p_obs).shape == (minibatch_size, n_actions)


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
    p_obs = create_p_obs(1)[0]
    action = 2
    rew = 50
    p_obs2 = 2 * p_obs
    done = False

    m = ReplayMemory(10, p_obs.shape)
    m.store(p_obs, action, rew, p_obs2, done)

    po_s, a_s, r_s, po2_s, d_s = m.sample(1)
    # import pdb; pdb.set_trace()
    assert (po_s[0] == p_obs).all()
    assert a_s[0] == action
    assert r_s[0] == rew
    assert (po2_s[0] == p_obs2).all()
    assert d_s[0] == done

    # assert po_s[0] == np.expand_dims(p_obs, 0)).all()
    # assert a_s == np.expand_dims(action, 0)).all()
    # assert r_s == np.expand_dims(rew, 0)).all()
    # assert po2_s == np.expand_dims(p_obs2, 0)).all()
    # assert po2_s == np.expand_dims(p_obs2, 0

