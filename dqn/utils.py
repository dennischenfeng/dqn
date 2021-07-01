import warnings
import numpy as np
import torch as th
import torch.nn as nn
from torchvision import transforms
import datetime

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


def evaluate_model(model, env, num_episodes=10, max_steps=int(1e6)):
    with th.no_grad():
        ep_rews = []
        obs = env.reset()
        for _ in range(num_episodes):
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

    return ep_rews


def basic_mlp_network(n_inputs, n_outputs):
    net = nn.Sequential(
        nn.Linear(n_inputs, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, n_outputs)
    )
    return net


def datetime_string():
    return (datetime.datetime.now()).strftime('%Y%m%d-%H%M%S')
