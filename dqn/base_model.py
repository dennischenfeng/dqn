"""
Base model for RL algorithm
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """
    Base model for RL algorithm
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(self, obs):
        """
        Use the model to predict an action
        """
        pass
