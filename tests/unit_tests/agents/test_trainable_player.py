#
# A pokémon showdown battle-bot project based on reinforcement learning techniques.
# Copyright (C) 2022 Matteo Dell'Acqua
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import copy
import pytest

from typing import List, Callable, Tuple
from unittest.mock import MagicMock, patch

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder
from poke_env.player.player import Player

from agents import (
    EPSILON_WHILE_TRAINING_AND_PLAYING,
    MIN_EPSILON_WHILE_TRAINING,
    LEARNING_RATE_WHILE_PLAYING,
    MIN_LEARNING_RATE_WHILE_TRAINING,
)
from agents.base_classes.trainable_player import TrainablePlayer


class DummyTrainablePlayer(TrainablePlayer):
    def _get_battle_to_state_func(self) -> Callable[[AbstractBattle], Tuple[float]]:
        pass

    def _get_action_to_move_func(self) -> Callable[[Player, int, Battle], BattleOrder]:
        pass

    def _get_action_space_size(self) -> int:
        return 2

    def _train(self, last_state, last_action, reward) -> None:
        pass

    @staticmethod
    def _calc_reward(
        last_battle: AbstractBattle, current_battle: AbstractBattle
    ) -> float:
        pass

    def _state_headers(self) -> List[str]:
        pass

    def _action_space_headers(self) -> List[str]:
        pass


def test_init_method():
    agent = DummyTrainablePlayer(
        start_listening=False,
        battle_format="gen8randombattle",
        model=None,
        training=False,
        keep_training=False,
    )
    assert isinstance(agent, TrainablePlayer)
    assert agent.action_space_size == 2
    assert not agent.training
    assert not agent.train_while_playing
    assert agent.model == {}


def test_choose_move_not_battle():
    battle = "test"
    agent = DummyTrainablePlayer(start_listening=False)
    with pytest.raises(RuntimeError):
        agent.choose_move(battle)  # noqa


def test_choose_move_no_training():
    battle = Battle("battle_tag", "username", None)  # noqa
    agent = DummyTrainablePlayer(start_listening=False)
    agent._calc_reward = MagicMock()
    agent._calc_reward.return_value = 0.0
    agent._train = MagicMock()
    agent._battle_to_state = MagicMock()
    state = (1, 0)
    agent._battle_to_state.return_value = state
    agent._choose_action = MagicMock()
    agent._choose_action.return_value = 1
    agent._action_to_move = MagicMock()
    order = ForfeitBattleOrder()
    agent._action_to_move.return_value = order
    agent._copy_battle = MagicMock()

    assert agent.choose_move(battle) is order
    agent._calc_reward.assert_not_called()
    agent._train.assert_not_called()
    agent._battle_to_state.assert_called_once_with(battle)
    agent._choose_action.assert_called_once_with(state)
    agent._copy_battle.assert_not_called()
    agent._action_to_move.assert_called_once_with(1, battle)


def test_choose_move_train():
    battle = Battle("battle_tag", "username", None)  # noqa
    agent = DummyTrainablePlayer(start_listening=False, training=True)
    agent._calc_reward = MagicMock()
    agent._calc_reward.return_value = 0.0
    agent._train = MagicMock()
    agent._battle_to_state = MagicMock()
    state = (1, 0)
    agent._battle_to_state.return_value = state
    agent._choose_action = MagicMock()
    agent._choose_action.return_value = 1
    agent._action_to_move = MagicMock()
    order = ForfeitBattleOrder()
    agent._action_to_move.return_value = order
    agent._copy_battle = MagicMock()
    battle_copy = copy.deepcopy(battle)
    agent._copy_battle.return_value = battle_copy

    assert agent.choose_move(battle) is order
    agent._calc_reward.assert_not_called()
    agent._train.assert_not_called()
    agent._battle_to_state.assert_called_once_with(battle)
    agent._choose_action.assert_called_once_with(state)
    agent._copy_battle.assert_called_once_with(battle)
    assert agent.last_state == battle_copy
    assert agent.last_action == 1
    agent._action_to_move.assert_called_once_with(1, battle)

    agent._calc_reward.reset_mock()
    agent._train.reset_mock()
    agent._battle_to_state.reset_mock()
    agent._choose_action.reset_mock()
    agent._action_to_move.reset_mock()
    agent._copy_battle.reset_mock()

    order = ForfeitBattleOrder()
    agent._action_to_move.return_value = order

    assert agent.choose_move(battle) is order
    agent._calc_reward.assert_called_once_with(battle_copy, battle)
    agent._train.assert_called_once_with(state, 1, 0.0)
    agent._battle_to_state.assert_called_with(battle)
    agent._choose_action.assert_called_with(state)
    agent._copy_battle.assert_called_once_with(battle)
    assert agent.last_state == battle_copy
    assert agent.last_action == 1
    agent._action_to_move.assert_called_once_with(1, battle)


def test_choose_action_new_state():
    agent = DummyTrainablePlayer(start_listening=False, training=True)
    agent._get_epsilon = MagicMock()
    agent._get_epsilon.return_value = 0.0
    model = {(1, 0): [[1, 2], 3, [2, 1]]}
    agent.model = model
    agent._choose_action((0, 1))
    assert agent.model == {(1, 0): [[1, 2], 3, [2, 1]], (0, 1): [[0, 0], 0, [0, 0]]}


def test_choose_action_even():
    agent = DummyTrainablePlayer(start_listening=False, training=True)
    agent._get_epsilon = MagicMock()
    agent._get_epsilon.return_value = 0.0
    model = {(1, 0): [[0, 0], 3, [2, 1]]}
    agent.model = model
    results = [0, 0]
    for _ in range(1_000_000):
        results[agent._choose_action((1, 0))] += 1
    results = [res / 1_000_000 for res in results]
    for res in results:
        assert 0.5 - res < 0.005


def test_choose_action_clear():
    agent = DummyTrainablePlayer(start_listening=False, training=True)
    agent._get_epsilon = MagicMock()
    agent._get_epsilon.return_value = 0.0
    model = {(1, 0): [[1, 0], 3, [2, 1]]}
    agent.model = model
    results = [0, 0]
    for _ in range(1000):
        results[agent._choose_action((1, 0))] += 1
    assert results[0] == 1000 and results[1] == 0


def test_choose_action_epsilon():
    agent = DummyTrainablePlayer(start_listening=False, training=True)
    agent._get_epsilon = MagicMock()
    agent._get_epsilon.return_value = 1.0
    model = {(1, 0): [[1, 0], 3, [2, 1]]}
    agent.model = model
    results = [0, 0]
    for _ in range(1000):
        results[agent._choose_action((1, 0))] += 1
    assert results[0] == 0 and results[1] == 1000


def test_action_to_move():
    battle = Battle("battle_tag", "username", None)  # noqa
    agent = DummyTrainablePlayer(start_listening=False, training=True)
    agent.action_to_move_function = MagicMock()
    return_string = "test"
    agent.action_to_move_function.return_value = return_string
    assert agent._action_to_move(1, battle) == return_string
    agent.action_to_move_function.assert_called_once_with(agent, 1, battle)


def test_battle_to_state():
    battle = Battle("battle_tag", "username", None)  # noqa
    agent = DummyTrainablePlayer(start_listening=False, training=True)
    agent.battle_to_state_func = MagicMock()
    return_string = "test"
    agent.battle_to_state_func.return_value = return_string
    assert agent._battle_to_state(battle) == return_string
    agent.battle_to_state_func.assert_called_with(battle)


def test_get_epsilon_fixed():
    agent = DummyTrainablePlayer(start_listening=False, training=False)
    for i in range(1000):
        assert agent._get_epsilon(2**i) == 0
    agent = DummyTrainablePlayer(start_listening=False, keep_training=True)
    for i in range(1000):
        assert agent._get_epsilon(2**i) == EPSILON_WHILE_TRAINING_AND_PLAYING


def test_get_epsilon_train():
    agent = DummyTrainablePlayer(start_listening=False, training=True)
    last = 1.0
    for i in range(100_000):
        current = agent._get_epsilon(i)
        assert current <= last
        assert current <= 1.0
        assert current >= MIN_EPSILON_WHILE_TRAINING
        last = current


def test_get_lr_fixed():
    agent = DummyTrainablePlayer(start_listening=False, training=False)
    for i in range(1000):
        assert agent._get_learning_rate(2**i) == 0
    agent = DummyTrainablePlayer(start_listening=False, keep_training=True)
    for i in range(1000):
        assert agent._get_learning_rate(2**i) == LEARNING_RATE_WHILE_PLAYING


def test_get_lr_train():
    agent = DummyTrainablePlayer(start_listening=False, training=True)
    last = 1.0
    for i in range(100_000):
        current = agent._get_learning_rate(i)
        assert current <= last
        assert current <= 1.0
        assert current >= MIN_LEARNING_RATE_WHILE_TRAINING
        last = current


def test_battle_finished_callback_no_train():
    battle = Battle("battle_tag", "username", None)  # noqa
    agent = DummyTrainablePlayer(start_listening=False)
    agent._calc_reward = MagicMock()
    agent._battle_to_state = MagicMock()
    agent._train = MagicMock()

    agent._battle_finished_callback(battle)
    agent._train.assert_not_called()
    agent._battle_to_state.assert_not_called()
    agent._calc_reward.assert_not_called()


def test_battle_finished_callback_train_battle_not_finished():
    battle = Battle("battle_tag", "username", None)  # noqa
    battle._finished = False
    other_battle = Battle("1", "1", None)  # noqa
    agent = DummyTrainablePlayer(start_listening=False, training=True)
    agent.last_state = other_battle
    agent._calc_reward = MagicMock()
    agent._calc_reward.return_value = 0.5
    agent._battle_to_state = MagicMock()
    agent._train = MagicMock()

    with pytest.raises(RuntimeError):
        agent._battle_finished_callback(battle)
    agent._train.assert_not_called()
    agent._battle_to_state.assert_not_called()
    agent._calc_reward.assert_called_once_with(other_battle, battle)


def test_battle_finished_callback_train_battle_finished():
    battle = Battle("battle_tag", "username", None)  # noqa
    battle._finished = True
    other_battle = Battle("1", "1", None)  # noqa
    agent = DummyTrainablePlayer(start_listening=False, training=True)
    agent.last_state = other_battle
    agent.last_action = 2
    agent._calc_reward = MagicMock()
    agent._calc_reward.return_value = 0.5
    agent._battle_to_state = MagicMock()
    state = (1, 2)
    agent._battle_to_state.return_value = state
    agent._train = MagicMock()

    agent._battle_finished_callback(battle)
    agent._calc_reward.assert_called_once_with(other_battle, battle)
    agent._battle_to_state.assert_called_once_with(other_battle)
    agent._train.assert_called_once_with(state, 2, 0.5)
    assert agent.last_state is None
    assert agent.last_action is None


def test_copy_battle():
    battle = Battle("battle_tag", "username", None)  # noqa
    agent = DummyTrainablePlayer(start_listening=False)
    with patch("copy.deepcopy") as mock_copy:
        mock_copy.return_value = 42
        assert agent._copy_battle(battle) == 42
        mock_copy.assert_called_once_with(battle)


def test_get_model():
    agent = DummyTrainablePlayer(start_listening=False)
    model = ForfeitBattleOrder()
    agent.model = model
    assert agent.get_model() is model


def test_get_pretty_model():
    agent = DummyTrainablePlayer(start_listening=False)
    model = ForfeitBattleOrder()
    agent.model = model
    agent._model_to_table = MagicMock()
    agent._model_to_table.return_value = 42
    assert agent.get_pretty_model() == 42
    agent._model_to_table.assert_called_once_with(model)


def test_reset_rates():
    agent = DummyTrainablePlayer(start_listening=False)
    model = {(1, 0): [[1, 0], 3, [2, 1]], (0, 1): [[0, 1], 5, [3, 2]]}
    agent.model = model
    agent.reset_rates()
    assert agent.model == {(1, 0): [[1, 0], 0, [2, 1]], (0, 1): [[0, 1], 0, [3, 2]]}
