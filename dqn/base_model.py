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
    def learn(self, *args, **kwargs) -> Any:
        """
        Train the model
        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """
        Use the model to predict an action
        """
        pass
