# Base class for a DQN Player
from abc import ABC, abstractmethod

from agents.base_classes.tf_player import TFPlayer


class DQNPlayer(TFPlayer, ABC):
    def train(self, num_iterations: int):
        pass

    @abstractmethod
    def eval_function(self, *args, **kwargs):
        pass

    @abstractmethod
    def log_function(self, *args, **kwargs):
        pass
