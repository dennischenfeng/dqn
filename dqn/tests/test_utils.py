"""
Test utils module
"""

import pytest
import gym
import torch as th
from dqn.utils import SimpleCrop, annealed_epsilon, evaluate_model, basic_mlp_network
from dqn.dqn import DQN


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


def test_annealed_epsilon():
    assert annealed_epsilon(0, 1, 0.1, 50) == pytest.approx(1)
    assert annealed_epsilon(25, 1, 0.1, 50) == pytest.approx(0.55)
    assert annealed_epsilon(50, 1, 0.1, 50) == pytest.approx(0.1)
    assert annealed_epsilon(100, 1, 0.1, 50) == pytest.approx(0.1)

    assert annealed_epsilon(100, 1, 0.1, 200) == pytest.approx(0.55)
    assert annealed_epsilon(100, 1, 0, 200) == pytest.approx(0.5)
    assert annealed_epsilon(100, 0.5, 0, 200) == pytest.approx(0.25)


def test_evaluate_model():
    env = gym.make("CartPole-v1")
    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    q_network = basic_mlp_network(n_inputs, n_actions)

    model = DQN(env, q_network=q_network, replay_memory_size=100)

    res = evaluate_model(model, env, num_episodes=2, max_steps=500)
    assert isinstance(res, list)

    with pytest.warns(UserWarning):
        res = evaluate_model(model, env, num_episodes=2, max_steps=1)
        assert isinstance(res, list)
