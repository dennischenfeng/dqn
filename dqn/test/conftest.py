import pytest
import numpy as np


@pytest.fixture(scope="session")
def mod_obs_batch():
    # minibatch size 32, 4 channels, 84 x 84 image
    return np.ones((32, 4, 84, 84))