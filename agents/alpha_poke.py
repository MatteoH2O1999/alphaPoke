# Module containing production-level agents with neural networks
from abc import ABC
from gym.spaces import Space
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.baselines import (
    RandomPlayer,
    MaxBasePowerPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.player.openai_api import ObservationType
from poke_env.player.player import Player
from poke_env.player.utils import background_evaluate_player
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
    def __init__(self, log_interval=1000, eval_interval=10_000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_int = log_interval
        self.eval_int = eval_interval

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

    @property
    def log_interval(self) -> int:
        return self.log_int

    @property
    def eval_interval(self) -> int:
        return self.eval_int

    def eval_function(self, step):
        num_challenges = 2000
        placement_challenges = 40

        eval_env, agent = self.create_evaluation_env(active=False)
        policy = self.agent.policy
        task = background_evaluate_player(
            agent, n_battles=num_challenges, n_placement_battles=placement_challenges
        )
        for _ in range(num_challenges):
            time_step = eval_env.reset()
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step)
        evaluation = task.result()

        if "evaluations" not in self.evaluations.keys():
            self.evaluations["evaluations"] = [[], []]
        self.evaluations["evaluations"][0].append(step)
        self.evaluations["evaluations"][1].append(evaluation)

    def log_function(self, step, loss_info: LossInfo):
        if "losses" not in self.evaluations.keys():
            self.evaluations["losses"] = [[], []]
        self.evaluations["losses"][0].append(step)
        self.evaluations["losses"][1].append(loss_info.loss)


class AlphaPokeDQN(AlphaPokeEmbedded):
    def get_agent(self) -> TFAgent:
        pass

    def get_replay_buffer(self) -> ReplayBuffer:
        pass

    def get_replay_buffer_iterator(self) -> Iterator:
        pass

    def get_collect_driver(self) -> PyDriver:
        pass
