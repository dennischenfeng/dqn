import pytest
import numpy as np
import torch as th
from dqn.dqn import QNetwork, ReplayMemory, annealed_epsilon


def generate_p_obs(minibatch_size):
    # minibatch size 32, 4 channels, 84 x 84 image
    return th.zeros((minibatch_size, 4, 84, 84))


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
            p_obs = generate_p_obs(minibatch_size)
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
    assert (o_s == np.expand_dims(np.ones(5), 0)).all()
    assert (a_s == np.expand_dims(1, 0)).all()
    assert (r_s == np.expand_dims(100, 0)).all()
    assert (o2_s == np.expand_dims(2 * np.ones(5), 0)).all()
    assert (d_s == np.expand_dims(False, 0)).all()

    with pytest.raises(ValueError):
        m.sample(2)

    # Store 10 total (9 more)
    for _ in range(9):
        o, a, r, o2, d = next(g)
        m.store(o, a, r, o2, d)

    o_s, a_s, r_s, o2_s, d_s = m.sample(10)
    assert 1 in a_s  # Sample 1 is still here
    assert 100 in r_s

    assert m.num_stores == 10
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


