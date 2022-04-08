# Module containing production-level agents with neural networks
from abc import ABC
from gym import Space
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.baselines import (
    RandomPlayer,
    MaxBasePowerPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.player.openai_api import ObservationType
from poke_env.player.player import Player
from tf_agents.agents import TFAgent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.drivers.py_driver import PyDriver
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from typing import Iterator, Union, List

from agents.base_classes.dqn_player import DQNPlayer
from agents.seba import Seba

rewards = {
    "fainted_value": 0.0,
    "hp": 0.0,
    "number_of_pokemons": 6,
    "starting_value": 0.0,
    "victory_reward": 1.0,
}


class AlphaPokeEmbedded(DQNPlayer, ABC):
    def calc_reward(
        self, last_battle: AbstractBattle, current_battle: AbstractBattle
    ) -> float:
        return self.reward_computing_helper(
            current_battle,
            fainted_value=rewards["fainted"],
            hp_value=rewards["hp"],
            number_of_pokemons=rewards["number_of_pokemons"],
            starting_value=rewards["starting_value"],
            status_value=rewards["status_value"],
            victory_value=rewards["victory_reward"],
        )

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        pass

    @property
    def embedding(self) -> Space:
        pass

    @property
    def opponents(self) -> Union[Player, str, List[Player], List[str]]:
        opponents_classes = [
            RandomPlayer,
            MaxBasePowerPlayer,
            SimpleHeuristicsPlayer,
            Seba,
        ]
        return [cls(battle_format=self.battle_format) for cls in opponents_classes]


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
