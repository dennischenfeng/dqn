import warnings
import numpy as np
import torch as th
from torchvision import transforms


class SimpleCrop(th.nn.Module):
    """
    Crops an image (deterministically) using the TF.crop function. (No simple crop can be found in the
    torchvision.transforms library
    """
    def __init__(self, i, j, h, w):
        super().__init__()
        self.i = i
        self.j = j
        self.h = h
        self.w = w

    def forward(self, img):
        return transforms.functional.crop(img, self.i, self.j, self.h, self.w)


def annealed_epsilon(step, epsilon_start, epsilon_stop, anneal_finished_step):
    return epsilon_start + (epsilon_stop - epsilon_start) * min(1, step / anneal_finished_step)


def evaluate_model(model, env, num_trials=10, max_steps=int(1e6)):
    with th.no_grad():
        ep_rews = []
        obs = env.reset()
        for i in range(num_trials):
            ep_rew = 0
            done = False
            for step in range(max_steps):
                action = model.predict(obs)
                obs, reward, done, info = env.step(action)
                ep_rew += reward
                if done:
                    obs = env.reset()
                    break

            ep_rews.append(ep_rew)

            if not done:
                warnings.warn(f"While evaluating the model, reached max_steps ({max_steps}) before reaching terminal "
                              f"state in env. Terminating it at max_steps.")

    return np.mean(ep_rews)
