import pytest
import torch as th
from dqn.dqn import QNetwork, annealed_epsilon


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


