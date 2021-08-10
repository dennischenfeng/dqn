import gym
import pytest
import numpy as np
from torchvision import transforms
import mock
from dqn.preprocessed_atari_env import preprocess_obs_maxed_seq, create_preprocessing_transform, ATARI_OBS_SHAPE, \
    OBS_SEQUENCE_LENGTH, PreprocessedAtariEnv


@pytest.fixture(scope="function")
def obs_maxed_seq():
    obs_maxed_seq = [np.ones(ATARI_OBS_SHAPE)] * OBS_SEQUENCE_LENGTH
    return np.array(obs_maxed_seq)


@pytest.fixture(scope="function")
def mock_pong_env():
    def mock_step_values():
        i = 1
        for _ in range(10):
            obs2 = i * np.ones(ATARI_OBS_SHAPE)
            obs2[0, 0, 0] = -i
            rew = i * 0.11
            done = (i % 10 == 0)
            info = {"ale.lives": 0}
            yield obs2, rew, done, info
            i += 1

    # Convert to list b/c deepcopy (from determining initial_num_lives) doesn't work on generators
    mock_step_values_list = list(mock_step_values())

    mock_env = mock.MagicMock(spec=gym.Env)
    mock_env.step.side_effect = mock_step_values_list
    mock_env.reset.return_value = np.zeros(ATARI_OBS_SHAPE)
    mock_env.action_space = gym.spaces.discrete.Discrete(1)
    mock_env.spec.id = "PongNoFrameskip-v4"

    return mock_env


def test_preprocess_obs_maxed_seq(obs_maxed_seq):
    preprocess_transform = create_preprocessing_transform(0)
    mod_obs = preprocess_obs_maxed_seq(obs_maxed_seq, preprocess_transform)
    assert mod_obs.shape == (4, 84, 84)


def test_reset(mock_pong_env):
    env = PreprocessedAtariEnv(mock_pong_env)
    mod_obs = env.reset()
    assert (np.array(mod_obs) == np.zeros((4, 84, 84))).all()


@mock.patch('dqn.preprocessed_atari_env.preprocess_obs_maxed_seq')
def test_step(mock_preprocess, mock_pong_env):
    def mock_preprocess_side_effect(obs_maxed_seq, preprocess_transform, device=None):
        return obs_maxed_seq
    mock_preprocess.side_effect = mock_preprocess_side_effect

    env = PreprocessedAtariEnv(mock_pong_env, clip_reward=True)
    env.reset()

    mod_obs, rew, done, info = env.step(0)
    assert mock_pong_env.step.call_count == 4
    assert rew == pytest.approx(0.11 + 0.22 + 0.33 + 0.44)
    assert not done
    assert info == {"ale.lives": 0}
    # took element-wise maximum correctly
    assert mod_obs[-1][0, 0, 0] == -3
    assert mod_obs[-1][0, 0, 1] == 4

    mod_obs, rew, done, info = env.step(0)
    assert mock_pong_env.step.call_count == 8
    assert rew == pytest.approx(0.55 + 0.66 + 0.77 + 0.88)
    assert not done
    assert info == {"ale.lives": 0}
    # took element-wise maximum correctly
    assert mod_obs[-1][0, 0, 0] == -7
    assert mod_obs[-1][0, 0, 1] == 8

    # Env reaches terminal state before all action_repeat steps
    mod_obs, rew, done, info = env.step(0)
    assert mock_pong_env.step.call_count == 10
    # reward is clipped at 1
    assert rew == pytest.approx(0.99 + 1.0)
    assert done
    assert info == {"ale.lives": 0}
    # took element-wise maximum correctly
    assert mod_obs[-1][0, 0, 0] == -9
    assert mod_obs[-1][0, 0, 1] == 10


@mock.patch('dqn.preprocessed_atari_env.preprocess_obs_maxed_seq')
def test_step_no_clip_reward(mock_preprocess, mock_pong_env):
    def mock_preprocess_side_effect(obs_maxed_seq, preprocess_transform, device=None):
        return obs_maxed_seq
    mock_preprocess.side_effect = mock_preprocess_side_effect

    env = PreprocessedAtariEnv(mock_pong_env, clip_reward=False)
    env.reset()

    mod_obs, rew, done, info = env.step(0)
    assert mock_pong_env.step.call_count == 4
    assert rew == pytest.approx(0.11 + 0.22 + 0.33 + 0.44)

    mod_obs, rew, done, info = env.step(0)
    assert mock_pong_env.step.call_count == 8
    assert rew == pytest.approx(0.55 + 0.66 + 0.77 + 0.88)

    # Env reaches terminal state before all action_repeat steps
    mod_obs, rew, done, info = env.step(0)
    assert mock_pong_env.step.call_count == 10
    # reward is not clipped
    assert rew == pytest.approx(0.99 + 1.1)