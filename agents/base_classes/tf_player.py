# Base class for a trainable player using TF-Agents.
import asyncio
import os
import tensorflow as tf

from abc import ABC, abstractmethod
from asyncio import Event
from gym import Space
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.openai_api import OpenAIGymEnv, ObservationType
from poke_env.player.player import Player
from tf_agents.agents import TFAgent
from tf_agents.drivers.py_driver import PyDriver
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.policies import TFPolicy, policy_saver
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from typing import Awaitable, Callable, Iterator, List, Optional, Union

from utils.action_to_move_function import (
    get_int_action_to_move,
    get_int_action_space_size,
)


class _Env(OpenAIGymEnv):
    def __init__(
        self,
        username: str,
        calc_reward: Callable[[AbstractBattle, AbstractBattle], float],
        action_to_move: Callable[[Player, int, AbstractBattle], BattleOrder],
        embed_battle: Callable[[AbstractBattle], ObservationType],
        embedding_description: Space,
        action_space_size: int,
        opponents: Union[Player, str, List[Player], List[str]],
        *args,
        **kwargs,
    ):
        self.calc_reward_func = calc_reward
        self.action_to_move_func = action_to_move
        self.embed_battle_func = embed_battle
        self.embedding_description = embedding_description
        self.space_size = action_space_size
        self.opponents = opponents
        tmp = self.__class__.__name__
        self.__class__.__name__ = username
        super().__init__(*args, **kwargs)
        self.__class__.__name__ = tmp

    def calc_reward(
        self, last_battle: AbstractBattle, current_battle: AbstractBattle
    ) -> float:
        return self.calc_reward_func(last_battle, current_battle)

    def action_to_move(self, action: int, battle: AbstractBattle) -> BattleOrder:
        return self.action_to_move_func(self.agent, action, battle)

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        return self.embed_battle_func(battle)

    def describe_embedding(self) -> Space:
        return self.embedding_description

    def action_space_size(self) -> int:
        return self.space_size

    def get_opponent(self) -> Union[Player, str, List[Player], List[str]]:
        return self.opponents


class TFPlayer(Player, ABC):
    def __init__(  # noqa: super().__init__ won't get called as this is a "fake" Player class
        self, model: str = None, *args, **kwargs
    ):
        self.battle_format = kwargs.get("battle_format", "gen8randombattle")
        kwargs["start_challenging"] = False
        temp_env = _Env(
            self.__class__.__name__,
            self.calc_reward_func,
            self.action_to_move_func,
            self.embed_battle_func,
            self.embedding,
            self.space_size,
            self.opponents if model is None else None,
            *args,
            **kwargs,
        )
        self.internal_agent = temp_env.agent
        self.wrapped_env = temp_env
        temp_env = suite_gym.wrap_env(temp_env)
        self.environment = tf_py_environment.TFPyEnvironment(temp_env)
        self._reward_buffer = {}
        self.agent: TFAgent
        self.policy: TFPolicy
        self.replay_buffer: ReplayBuffer
        self.replay_buffer_iterator: Iterator
        self.collect_driver: PyDriver
        if model is None:
            self.can_train = True
            self.evaluations = {}
            self.agent = self.get_agent()
            self.policy = self.agent.policy
            self.replay_buffer = self.get_replay_buffer()
            self.replay_buffer_iterator = self.get_replay_buffer_iterator()
            self.collect_driver = self.get_collect_driver()
            self.saver = policy_saver.PolicySaver(self.agent.policy)
        else:
            if not os.path.isdir(model):
                raise ValueError("Expected directory as model parameter.")
            if not tf.saved_model.contains_saved_model(model):
                raise ValueError("Expected saved model as model parameter.")
            self.can_train = False
            self.policy = tf.saved_model.load(model)
        if not isinstance(self.policy, TFPolicy):
            raise RuntimeError(
                f"Expected subclass of TFPolicy, got {type(self.policy)}"
            )

    @property
    def calc_reward_func(self) -> Callable[[AbstractBattle, AbstractBattle], float]:
        return self.calc_reward

    @abstractmethod
    def calc_reward(
        self, last_battle: AbstractBattle, current_battle: AbstractBattle
    ) -> float:  # pragma: no cover
        pass

    @property
    def embed_battle_func(self) -> Callable[[AbstractBattle], ObservationType]:
        return self.embed_battle

    @abstractmethod
    def embed_battle(
        self, battle: AbstractBattle
    ) -> ObservationType:  # pragma: no cover
        pass

    @property
    @abstractmethod
    def embedding(self) -> Space:  # pragma: no cover
        pass

    @property
    @abstractmethod
    def opponents(
        self,
    ) -> Union[Player, str, List[Player], List[str]]:  # pragma: no cover
        pass

    @abstractmethod
    def get_agent(self) -> TFAgent:  # pragma: no cover
        pass

    @abstractmethod
    def get_replay_buffer(self) -> ReplayBuffer:  # pragma: no cover
        pass

    @abstractmethod
    def get_replay_buffer_iterator(self) -> Iterator:  # pragma: no cover
        pass

    @abstractmethod
    def get_collect_driver(self) -> PyDriver:  # pragma: no cover
        pass

    @abstractmethod
    def log_function(self, *args, **kwargs):  # pragma: no cover
        pass

    @abstractmethod
    def eval_function(self, *args, **kwargs):  # pragma: no cover
        pass

    @property
    @abstractmethod
    def log_interval(self) -> int:  # pragma: no cover
        pass

    @property
    @abstractmethod
    def eval_interval(self) -> int:  # pragma: no cover
        pass

    @abstractmethod
    def train(self, num_iterations: int):  # pragma: no cover
        pass

    def save_policy(self, save_dir):
        self.saver.save(save_dir)

    @property
    def action_to_move_func(
        self,
    ) -> Callable[[Player, int, AbstractBattle], BattleOrder]:
        format_lowercase = self.battle_format.lower()
        double = (
            "vgc" in format_lowercase
            or "double" in format_lowercase
            or "metronome" in format_lowercase
        )
        return get_int_action_to_move(self.battle_format, double)

    @property
    def space_size(self) -> int:
        format_lowercase = self.battle_format.lower()
        double = (
            "vgc" in format_lowercase
            or "double" in format_lowercase
            or "metronome" in format_lowercase
        )
        return get_int_action_space_size(self.battle_format, double)

    def create_evaluation_env(self, active=True):
        env = _Env(
            self.__class__.__name__,
            self.calc_reward_func,
            self.action_to_move_func,
            self.embed_battle_func,
            self.embedding,
            self.space_size,
            self.opponents if active else None,
            battle_format=self.battle_format,
            start_challenging=active,
        )
        agent = env.agent
        env = suite_gym.wrap_env(env)
        env = tf_py_environment.TFPyEnvironment(env)
        return env, agent

    def reward_computing_helper(
        self,
        battle: AbstractBattle,
        *,
        fainted_value: float = 0.0,
        hp_value: float = 0.0,
        number_of_pokemons: int = 6,
        starting_value: float = 0.0,
        status_value: float = 0.0,
        victory_value: float = 1.0,
    ) -> float:
        """A helper function to compute rewards.

        The reward is computed by computing the value of a game state, and by comparing
        it to the last state.

        State values are computed by weighting different factor. Fainted pokemons,
        their remaining HP, inflicted statuses and winning are taken into account.

        For instance, if the last time this function was called for battle A it had
        a state value of 8 and this call leads to a value of 9, the returned reward will
        be 9 - 8 = 1.

        Consider a single battle where each player has 6 pokemons. No opponent pokemon
        has fainted, but our team has one fainted pokemon. Three opposing pokemons are
        burned. We have one pokemon missing half of its HP, and our fainted pokemon has
        no HP left.

        The value of this state will be:

        - With fainted value: 1, status value: 0.5, hp value: 1:
            = - 1 (fainted) + 3 * 0.5 (status) - 1.5 (our hp) = -1
        - With fainted value: 3, status value: 0, hp value: 1:
            = - 3 + 3 * 0 - 1.5 = -4.5

        :param battle: The battle for which to compute rewards.
        :type battle: AbstractBattle
        :param fainted_value: The reward weight for fainted pokemons. Defaults to 0.
        :type fainted_value: float
        :param hp_value: The reward weight for hp per pokemon. Defaults to 0.
        :type hp_value: float
        :param number_of_pokemons: The number of pokemons per team. Defaults to 6.
        :type number_of_pokemons: int
        :param starting_value: The default reference value evaluation. Defaults to 0.
        :type starting_value: float
        :param status_value: The reward value per non-fainted status. Defaults to 0.
        :type status_value: float
        :param victory_value: The reward value for winning. Defaults to 1.
        :type victory_value: float
        :return: The reward.
        :rtype: float
        """
        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        for mon in battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value

        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value

        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value

        to_return = current_value - self._reward_buffer[battle]
        self._reward_buffer[battle] = current_value

        return to_return

    def choose_move(
        self, battle: AbstractBattle
    ) -> Union[BattleOrder, Awaitable[BattleOrder]]:
        """choose_move won't get implemented as this is a 'fake' Player class."""

    def play_episode(self):
        time_step = self.environment.reset()
        while not time_step.is_last():
            action_step = self.policy.action(time_step)
            time_step = self.environment.step(action_step.action)

    async def accept_challenges(
        self, opponent: Optional[Union[str, List[str]]], n_challenges: int
    ) -> None:  # pragma: no cover
        challenge_task = asyncio.ensure_future(
            self.internal_agent.accept_challenges(opponent, n_challenges)
        )
        for _ in range(n_challenges):
            while (
                self.internal_agent.current_battle is None
                or self.internal_agent.current_battle.finished
            ):
                await asyncio.sleep(0.1)
            self.play_episode()
        await challenge_task

    async def send_challenges(
        self, opponent: str, n_challenges: int, to_wait: Optional[Event] = None
    ) -> None:  # pragma: no cover
        challenge_task = asyncio.ensure_future(
            self.internal_agent.send_challenges(opponent, n_challenges, to_wait)
        )
        for _ in range(n_challenges):
            while (
                self.internal_agent.current_battle is None
                or self.internal_agent.current_battle.finished
            ):
                await asyncio.sleep(0.1)
            self.play_episode()
        await challenge_task

    async def battle_against(
        self, opponent: Player, n_battles: int
    ) -> None:  # pragma: no cover
        challenge_task = asyncio.ensure_future(
            self.internal_agent.battle_against(opponent, n_battles)
        )
        for _ in range(n_battles):
            while (
                self.internal_agent.current_battle is None
                or self.internal_agent.current_battle.finished
            ):
                await asyncio.sleep(0.1)
            self.play_episode()
        await challenge_task

    async def ladder(self, n_games):  # pragma: no cover
        challenge_task = asyncio.ensure_future(self.internal_agent.ladder(n_games))
        for _ in range(n_games):
            while (
                self.internal_agent.current_battle is None
                or self.internal_agent.current_battle.finished
            ):
                await asyncio.sleep(0.1)
            self.play_episode()
        await challenge_task

    def __getattr__(self, item):
        return self.internal_agent.__getattribute__(item)
