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
import pytest

import unittest.mock

from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.player.random_player import RandomPlayer

from utils.action_to_move_function import (
    action_to_move_gen8single,
    get_int_action_to_move,
    get_int_action_space_size,
)
from utils.invalid_argument import InvalidAction


def get_mocks():
    mock_agent = unittest.mock.create_autospec(
        RandomPlayer, spec_set=True, instance=True
    )
    battle = Battle("battle_tag", "username", None)  # noqa
    return mock_agent, battle


# Test get int action to move


def test_get_int_action_to_move_gen8randombattle_success():
    assert (
        get_int_action_to_move("gen8randombattle", False) is action_to_move_gen8single
    )


def test_get_int_action_to_move_single_failure():
    with pytest.raises(NotImplementedError):
        get_int_action_to_move("noformat", False)


def test_get_int_action_to_move_double_failure():
    with pytest.raises(NotImplementedError):
        get_int_action_to_move("noformat", True)


# Test get int action space size


def test_get_int_action_space_size_gen8randombattle_success():
    assert get_int_action_space_size("gen8randombattle", False) == 22


def test_get_int_action_space_size_single_failure():
    with pytest.raises(NotImplementedError):
        get_int_action_space_size("noformat", False)


def test_get_int_action_space_size_double_failure():
    with pytest.raises(NotImplementedError):
        get_int_action_space_size("noformat", True)


# Generation 8 single battle tests


def test_forfeit():
    mock_agent, battle = get_mocks()
    assert isinstance(
        action_to_move_gen8single(mock_agent, -1, battle), ForfeitBattleOrder
    )


def test_gen8single_choose_normal_move_success():
    mock_agent, battle = get_mocks()
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 1, battle)
    mock_agent.create_order.assert_called_with(moves[1])


def test_gen8single_choose_normal_move_failure_out_of_range():
    mock_agent, battle = get_mocks()
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 2, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


def test_gen8single_choose_normal_move_failure_forced_switch():
    mock_agent, battle = get_mocks()
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    battle._force_switch = True
    action_to_move_gen8single(mock_agent, 1, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


@unittest.mock.patch(
    "poke_env.environment.pokemon.Pokemon.available_z_moves",
    new_callable=unittest.mock.PropertyMock,
)
def test_gen8single_choose_z_move_success(available_z_moves_mock):
    mock_agent, battle = get_mocks()
    battle._can_z_move = True
    active_pokemon = Pokemon(species="charizard")
    active_pokemon._active = True
    battle._team = {"charizerd": active_pokemon}
    z_move = Move("flamethrower")
    available_z_moves_mock.return_value = [z_move]
    action_to_move_gen8single(mock_agent, 4, battle)
    mock_agent.create_order.assert_called_with(z_move, z_move=True)


@unittest.mock.patch(
    "poke_env.environment.pokemon.Pokemon.available_z_moves",
    new_callable=unittest.mock.PropertyMock,
)
def test_gen8single_choose_z_move_failure_forced_switch(available_z_moves_mock):
    mock_agent, battle = get_mocks()
    battle._can_z_move = True
    battle._force_switch = True
    active_pokemon = Pokemon(species="charizard")
    active_pokemon._active = True
    battle._team = {"charizerd": active_pokemon}
    z_move = Move("flamethrower")
    available_z_moves_mock.return_value = [z_move]
    action_to_move_gen8single(mock_agent, 4, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


@unittest.mock.patch(
    "poke_env.environment.pokemon.Pokemon.available_z_moves",
    new_callable=unittest.mock.PropertyMock,
)
def test_gen8single_choose_z_move_failure_cannot_z_move(available_z_moves_mock):
    mock_agent, battle = get_mocks()
    battle._can_z_move = False
    active_pokemon = Pokemon(species="charizard")
    active_pokemon._active = True
    battle._team = {"charizerd": active_pokemon}
    z_move = Move("flamethrower")
    available_z_moves_mock.return_value = [z_move]
    action_to_move_gen8single(mock_agent, 4, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


@unittest.mock.patch(
    "poke_env.environment.pokemon.Pokemon.available_z_moves",
    new_callable=unittest.mock.PropertyMock,
)
def test_gen8single_choose_z_move_failure_no_active_pokemon(available_z_moves_mock):
    mock_agent, battle = get_mocks()
    battle._can_z_move = True
    active_pokemon = Pokemon(species="charizard")
    active_pokemon._active = False
    battle._team = {"charizerd": active_pokemon}
    z_move = Move("flamethrower")
    available_z_moves_mock.return_value = [z_move]
    action_to_move_gen8single(mock_agent, 4, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


@unittest.mock.patch(
    "poke_env.environment.pokemon.Pokemon.available_z_moves",
    new_callable=unittest.mock.PropertyMock,
)
def test_gen8single_choose_z_move_failure_out_of_range(available_z_moves_mock):
    mock_agent, battle = get_mocks()
    battle._can_z_move = True
    active_pokemon = Pokemon(species="charizard")
    active_pokemon._active = True
    battle._team = {"charizerd": active_pokemon}
    z_move = Move("flamethrower")
    available_z_moves_mock.return_value = [z_move]
    action_to_move_gen8single(mock_agent, 5, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


def test_gen8single_choose_mega_move_success():
    mock_agent, battle = get_mocks()
    battle._can_mega_evolve = True
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 9, battle)
    mock_agent.create_order.assert_called_with(moves[1], mega=True)


def test_gen8single_choose_mega_move_failure_cannot_mega_evolve():
    mock_agent, battle = get_mocks()
    battle._can_mega_evolve = False
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 9, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


def test_gen8single_choose_mega_move_failure_forced_switch():
    mock_agent, battle = get_mocks()
    battle._can_mega_evolve = True
    battle._force_switch = True
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 9, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


def test_gen8single_choose_mega_move_failure_out_of_range():
    mock_agent, battle = get_mocks()
    battle._can_mega_evolve = True
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 10, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


def test_gen8single_choose_dynamax_move_success():
    mock_agent, battle = get_mocks()
    battle._can_dynamax = True
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 13, battle)
    mock_agent.create_order.assert_called_with(moves[1], dynamax=True)


def test_gen8single_choose_dynamax_move_failure_cannot_dynamax():
    mock_agent, battle = get_mocks()
    battle._can_dynamax = False
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 13, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


def test_gen8single_choose_dynamax_move_failure_forced_switch():
    mock_agent, battle = get_mocks()
    battle._can_dynamax = True
    battle._force_switch = True
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 13, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


def test_gen8single_choose_dynamax_move_failure_out_of_range():
    mock_agent, battle = get_mocks()
    battle._can_dynamax = True
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 14, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


def test_gen8single_switch_success():
    mock_agent, battle = get_mocks()
    switches = [Pokemon(species="charizard"), Pokemon(species="pikachu")]
    battle._available_switches = switches
    action_to_move_gen8single(mock_agent, 17, battle)
    mock_agent.create_order.assert_called_with(switches[1])


def test_gen8single_switch_failure():
    mock_agent, battle = get_mocks()
    switches = [Pokemon(species="charizard"), Pokemon(species="pikachu")]
    battle._available_switches = switches
    action_to_move_gen8single(mock_agent, 18, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


def test_gen8single_failure_exception():
    mock_agent, battle = get_mocks()
    switches = [Pokemon(species="charizard"), Pokemon(species="pikachu")]
    battle._available_switches = switches
    with pytest.raises(InvalidAction):
        action_to_move_gen8single(mock_agent, 18, battle, InvalidAction)
    mock_agent.choose_random_move.assert_not_called()
