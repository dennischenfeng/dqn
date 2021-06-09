import gym

import torch as th
from torchvision import transforms

from dqn.utils import SimpleCrop, annealed_epsilon


ATARI_OBS_SHAPE = (210, 160, 3)
OBS_MAXED_SEQUENCE_LENGTH = 4  # number of obs_maxed's to keep as "last N frames" to feed as input to Q network
# Need different image cropping (roughly capturing the playing area of screen) for each env; starting row for crop
CROP_START_ROW = {"PongNoFrameskip-v4": 18}


class AtariEnv(gym.Env):
    def __init__(self, env, action_repeat=4):
        super().__init__()
        self.action_repeat = action_repeat

        # Need different image cropping (roughly capturing the playing area of screen) for each env
        game = env.spec.id
        crop_start_row = CROP_START_ROW[game]
        self.preprocess_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((110, 84)),
            SimpleCrop(crop_start_row, 0, 84, 84)  # TODO: confirm that Nature paper crops differently for each game
        ])

        # Instance variables for tracking state while stepping through env
        self.prev_obs = None
        self.latest_obs_maxed_seq = []

    def mod_env_reset(self):
        """

        :return:
        """
        obs = self.env.reset()
        # For first obs, maxing over current & "previous" is meaningless; use obs itself
        obs_maxed = obs
        self.latest_obs_maxed_seq = [obs_maxed] * OBS_MAXED_SEQUENCE_LENGTH

        # For next iteration
        self.prev_obs = obs

        mod_obs = self.preprocess_obs_maxed_seq()
        return mod_obs

    def mod_env_step(self, action):
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
            if done:
                break

            # For next iteration
            self.prev_obs = obs2

        self.latest_obs_maxed_seq.pop(0)
        self.latest_obs_maxed_seq.append(obs_maxed)
        mod_obs = self.preprocess_obs_maxed_seq()

        return mod_obs, total_rew, done, info

    def preprocess_obs_maxed_seq(self):
        assert len(self.latest_obs_maxed_seq) == OBS_MAXED_SEQUENCE_LENGTH
        for a in self.latest_obs_maxed_seq:
            assert a.shape == ATARI_OBS_SHAPE

        result = th.tensor(self.latest_obs_maxed_seq).float()
        result = result.permute(0, 3, 1, 2)
        result = self.preprocess_transform(result)
        # Squeeze out grayscale dimension (original RGB dim)
        result = result.squeeze(1)
        # TODO: remove
        assert tuple(result.shape) == (4, 84, 84)
        return result