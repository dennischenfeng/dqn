import gym
import torch as th
import numpy as np
from torchvision import transforms
from dqn.utils import SimpleCrop, annealed_epsilon
from copy import deepcopy

ATARI_OBS_SHAPE = (210, 160, 3)
MOD_OBS_SHAPE = (4, 84, 84)
OBS_MAXED_SEQUENCE_LENGTH = 4  # number of obs_maxed's to keep as "last N frames" to feed as input to Q network
# Need different image cropping (roughly capturing the playing area of screen) for each env; starting row for crop
CROP_START_ROW = {
    "PongNoFrameskip-v4": 18,
    "BreakoutNoFrameskip-v4": 18
}


class PreprocessedAtariEnv(gym.Env):
    def __init__(self, env, action_repeat=4):
        super().__init__()
        self.env = env
        self.action_repeat = action_repeat

        # Need different image cropping (roughly capturing the playing area of screen) for each env
        game = env.spec.id
        self.preprocess_transform = create_preprocessing_transform(CROP_START_ROW[game])

        # Instance variables for tracking state while stepping through env
        self.prev_obs = None
        self.latest_obs_maxed_seq = []

        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(0, 255, MOD_OBS_SHAPE, dtype=np.uint8)
        self.initial_num_lives = initial_num_lives(deepcopy(self.env))

    def reset(self):
        """

        :return:
        """
        obs = self.env.reset()
        # For first obs, maxing over current & "previous" is meaningless; use obs itself
        obs_maxed = obs
        self.latest_obs_maxed_seq = [obs_maxed] * OBS_MAXED_SEQUENCE_LENGTH

        # For next iteration
        self.prev_obs = obs

        mod_obs = preprocess_obs_maxed_seq(self.latest_obs_maxed_seq, self.preprocess_transform)
        return mod_obs

    def step(self, action):
        """

        :return:
        """
        obs_maxed = None
        total_rew = 0
        done = None
        info = None
        assert self.action_repeat >= 1
        for i in range(self.action_repeat):
            obs2, rew, done, info = self.env.step(action)

            obs_maxed = np.maximum(self.prev_obs, obs2)
            # As discussed in the paper, clip step rewards at -1 and +1 to limit scale of errors (potentially better
            # training stability), but reduces ability to differentiate actions for large/small rewards
            total_rew += float(np.clip(rew, -1, 1))

            # Losing a life terminates an episode
            if info["ale.lives"] != self.initial_num_lives:
                done = True

            if done:
                break

            # For next iteration
            self.prev_obs = obs2

        self.latest_obs_maxed_seq.pop(0)
        self.latest_obs_maxed_seq.append(obs_maxed)
        mod_obs = preprocess_obs_maxed_seq(self.latest_obs_maxed_seq, self.preprocess_transform)
        return mod_obs, total_rew, done, info

    def render(self, mode="human"):
        self.env.render(mode)


def preprocess_obs_maxed_seq(obs_maxed_seq, preprocess_transform):
    assert len(obs_maxed_seq) == OBS_MAXED_SEQUENCE_LENGTH
    for a in obs_maxed_seq:
        assert a.shape == ATARI_OBS_SHAPE

    # Numpy's conversion from list of arrays to array is much faster than pytorch's conversion to tensor
    obs_maxed_seq_arr = np.array(obs_maxed_seq)
    result = th.tensor(obs_maxed_seq_arr)

    result = result.permute(0, 3, 1, 2)
    result = preprocess_transform(result)
    # Squeeze out grayscale dimension (original RGB dim)
    result = result.squeeze(1)
    return np.array(result)


def create_preprocessing_transform(crop_start_row):
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((110, 84)),
        SimpleCrop(crop_start_row, 0, 84, 84)  # TODO: confirm that Nature paper crops differently for each game
    ])


def initial_num_lives(env):
    env.reset()
    action = env.action_space.sample()
    _, _, _, info = env.step(action)
    num_lives = info["ale.lives"]
    return num_lives


class ReorderedObsAtariEnv(gym.Env):
    def __init__(self, env, new_ordering=(2, 0, 1)):
        super().__init__()
        self.env = env
        self.new_ordering = new_ordering

        orig_obs_shape = self.env.observation_space.shape
        mod_obs_shape = tuple(np.array(orig_obs_shape)[self.new_ordering,])

        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(0, 255, mod_obs_shape, dtype=np.uint8)

    def reset(self):
        obs = self.env.reset()
        mod_obs = obs.transpose(*self.new_ordering)
        return mod_obs

    def step(self, action):
        obs2, rew, done, info = self.env.step(action)
        mod_obs2 = obs2.transpose(*self.new_ordering)
        return mod_obs2, rew, done, info

    def render(self, mode="human"):
        self.env.render(mode)
