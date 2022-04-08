# Module containing production-level agents with neural networks
from abc import ABC
from gym import Space
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.openai_api import ObservationType
from poke_env.player.player import Player
from tf_agents.agents import TFAgent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.drivers.py_driver import PyDriver
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from typing import Iterator, Union, List

from agents.base_classes.dqn_player import DQNPlayer


class AlphaPokeEmbedded(DQNPlayer, ABC):
    def calc_reward(
        self, last_battle: AbstractBattle, current_battle: AbstractBattle
    ) -> float:
        pass

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        pass

    @property
    def embedding(self) -> Space:
        pass

    @property
    def opponents(self) -> Union[Player, str, List[Player], List[str]]:
        pass


class AlphaPokeDQN(AlphaPokeEmbedded):
    def get_agent(self) -> TFAgent:
        pass

    def get_replay_buffer(self) -> ReplayBuffer:
        pass

    def get_replay_buffer_iterator(self) -> Iterator:
        pass

    def get_collect_driver(self) -> PyDriver:
        pass

    @property
    def log_interval(self) -> int:
        return 1000

    @property
    def eval_interval(self) -> int:
        return 10_000

    def eval_function(self, step):
        pass

    def log_function(self, step, loss_info: LossInfo):
        pass
