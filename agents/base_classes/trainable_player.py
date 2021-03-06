# General trainable state-based class with epsilon-greedy policy and variable learning rate
import copy
import math
import random

from abc import ABC, abstractmethod
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player
from typing import Tuple, Callable, List

from agents import (
    EPSILON_WHILE_TRAINING_AND_PLAYING,
    LEARNING_RATE_WHILE_PLAYING,
    MIN_EPSILON_WHILE_TRAINING,
    MIN_LEARNING_RATE_WHILE_TRAINING,
)
from utils import init_action_space


class TrainablePlayer(Player, ABC):
    def __init__(self, **kwargs):
        self.model = kwargs.get("model", {})
        if self.model is None:
            self.model = {}
        self.training = kwargs.get("training", False)
        self.train_while_playing = kwargs.get("keep_training", False)
        self.b_format = kwargs.get("battle_format")
        self.action_to_move_function = self._get_action_to_move_func()
        self.battle_to_state_func = self._get_battle_to_state_func()
        self.action_space_size = self._get_action_space_size()
        self.last_state = None
        self.last_action = None
        if "training" in kwargs.keys():
            kwargs.pop("training")
        if "model" in kwargs.keys():
            kwargs.pop("model")
        if "keep_training" in kwargs.keys():
            kwargs.pop("keep_training")
        if (
            self.training or self.train_while_playing
        ) and "max_concurrent_battles" in kwargs.keys():
            kwargs.pop("max_concurrent_battles")
        super().__init__(**kwargs)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if not isinstance(battle, Battle):
            raise RuntimeError("Error with battle transfer")
        if self.last_state and (self.training or self.train_while_playing):
            reward = self._calc_reward(self.last_state, battle)
            self._train(
                self._battle_to_state(self.last_state), self.last_action, reward
            )
        current_state = self._battle_to_state(battle)
        action = self._choose_action(current_state)
        if self.training or self.train_while_playing:
            self.last_state = self._copy_battle(battle)
            self.last_action = action
        return self._action_to_move(action, battle)

    def _choose_action(self, state):
        if state not in self.model.keys():
            self.model[state] = [
                init_action_space(self.action_space_size),
                0,
                init_action_space(self.action_space_size),
            ]
        value = self.model[state]
        action_space = value[0]
        epsilon = self._get_epsilon(value[1])
        max_value = max(action_space)
        optimal_actions = []
        for i, action in enumerate(action_space):
            if action == max_value:
                optimal_actions.append(i)
        if len(optimal_actions) > 1:
            optimal_action = random.choice(optimal_actions)
        else:
            optimal_action = optimal_actions[0]
        if random.random() < epsilon:
            tmp = optimal_action
            while optimal_action == tmp:
                optimal_action = random.randint(0, len(action_space) - 1)
        return optimal_action

    def _action_to_move(self, action: int, battle: Battle) -> BattleOrder:
        return self.action_to_move_function(self, action, battle)

    def _battle_to_state(self, battle: AbstractBattle):
        return self.battle_to_state_func(battle)

    @abstractmethod
    def _get_battle_to_state_func(
        self,
    ) -> Callable[[AbstractBattle], Tuple[float]]:  # pragma: no cover
        pass

    @abstractmethod
    def _get_action_to_move_func(
        self,
    ) -> Callable[[Player, int, Battle], BattleOrder]:  # pragma: no cover
        pass

    @abstractmethod
    def _get_action_space_size(self) -> int:  # pragma: no cover
        pass

    @abstractmethod
    def _train(self, last_state, last_action, reward) -> None:  # pragma: no cover
        pass

    @staticmethod
    @abstractmethod
    def _calc_reward(
        last_battle: AbstractBattle, current_battle: AbstractBattle
    ) -> float:  # pragma: no cover
        pass

    def _get_epsilon(self, samples):
        epsilon = 0
        if self.training:
            epsilon = max(
                min(1 / math.log(max(2, samples)), 1.0), MIN_EPSILON_WHILE_TRAINING
            )
        elif self.train_while_playing:
            epsilon = EPSILON_WHILE_TRAINING_AND_PLAYING
        return epsilon

    def _get_learning_rate(self, samples):
        learning_rate = 0
        if self.training:
            learning_rate = max(
                1.0, min(80 / max(1, samples), MIN_LEARNING_RATE_WHILE_TRAINING)
            )
        elif self.train_while_playing:
            learning_rate = LEARNING_RATE_WHILE_PLAYING
        return learning_rate

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        if self.training or self.train_while_playing:
            reward = self._calc_reward(self.last_state, battle)
            if not battle.finished:
                raise RuntimeError("???")
            if self.last_state and (self.training or self.train_while_playing):
                self._train(
                    self._battle_to_state(self.last_state), self.last_action, reward
                )
            self.last_state = None
            self.last_action = None

    @staticmethod
    def _copy_battle(battle):
        return copy.deepcopy(battle)

    def get_model(self):
        return self.model

    def get_pretty_model(self):
        return self._model_to_table(self.model)

    def reset_rates(self):
        for values in self.model.values():
            values[1] = 0

    def _model_to_table(self, model):  # pragma: no cover
        headers = []
        for h in self._state_headers():
            headers.append(h)
        for h in self._action_space_headers():
            headers.append(h)
        table = [headers]
        for state, actions in model.items():
            to_append = []
            for s in state:
                to_append.append(s)
            for a in actions[0]:
                to_append.append(a)
            table.append(to_append)
        return table

    @abstractmethod
    def _state_headers(self) -> List[str]:  # pragma: no cover
        pass

    @abstractmethod
    def _action_space_headers(self) -> List[str]:  # pragma: no cover
        pass
