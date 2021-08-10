"""
Test replay_memory module
"""

import pytest
import numpy as np
import gym
from dqn.replay_memory import ReplayMemory, initialize_replay_memory


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
    m = ReplayMemory(10, o.shape, o.dtype)

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
    m.sample(10)

    with pytest.raises(ValueError):
        m.sample(11)

    # Store 1 more
    o, a, r, o2, d = next(g)
    m.store(o, a, r, o2, d)
    m.sample(10)

    with pytest.raises(ValueError):
        m.sample(11)


def test_replay_memory_representative_obs():
    mod_obs = np.ones((4, 84, 84))
    action = 2
    rew = 50
    p_obs2 = 2 * mod_obs
    done = False

    m = ReplayMemory(10, mod_obs.shape, mod_obs.dtype)
    m.store(mod_obs, action, rew, p_obs2, done)

    pos_s, a_s, r_s, pos2_s, d_s = m.sample(1)
    assert (pos_s[0] == mod_obs).all()
    assert a_s[0] == action
    assert r_s[0] == rew
    assert (pos2_s[0] == p_obs2).all()
    assert d_s[0] == done


def test_initialize_replay_memory():
    env = gym.make("CartPole-v1")
    m = ReplayMemory(100, env.observation_space.shape, env.observation_space.dtype)

    n_steps = 70
    initialize_replay_memory(n_steps, env, m)
    assert m.num_stores == n_steps
