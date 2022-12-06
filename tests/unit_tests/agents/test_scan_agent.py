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
from unittest.mock import patch, MagicMock

from agents.utils.scan_agent import ScanAgent


def test_get_action_to_move_func_gen8randombattle():
    with patch("agents.utils.scan_agent.get_int_action_to_move") as mock_a2m:
        mock_func = MagicMock()
        mock_a2m.return_value = mock_func
        player = ScanAgent(start_listening=False, battle_format="gen8randombattle")
        func = player.get_action_to_move_func()
        assert func is mock_func
        mock_a2m.assert_called_once_with("gen8randombattle", False)


def test_get_action_to_move_func_gen8metronomebattle():
    with patch("agents.utils.scan_agent.get_int_action_to_move") as mock_a2m:
        mock_func = MagicMock()
        mock_a2m.return_value = mock_func
        player = ScanAgent(start_listening=False, battle_format="gen8metronomebattle")
        func = player.get_action_to_move_func()
        assert func is mock_func
        mock_a2m.assert_called_once_with("gen8metronomebattle", True)


def test_get_action_to_move_func_gen6ou():
    with patch("agents.utils.scan_agent.get_int_action_to_move") as mock_a2m:
        mock_func = MagicMock()
        mock_a2m.return_value = mock_func
        player = ScanAgent(start_listening=False, battle_format="gen6ou")
        func = player.get_action_to_move_func()
        assert func is mock_func
        mock_a2m.assert_called_once_with("gen6ou", False)


def test_action_to_move_func_gen8vgc2022():
    with patch("agents.utils.scan_agent.get_int_action_to_move") as mock_a2m:
        mock_func = MagicMock()
        mock_a2m.return_value = mock_func
        player = ScanAgent(start_listening=False, battle_format="gen8vgc2022")
        func = player.get_action_to_move_func()
        assert func is mock_func
        mock_a2m.assert_called_once_with("gen8vgc2022", True)


def test_choose_move_1():
    with patch("builtins.input") as mock_input:
        mock_input.return_value = 1
        mock_battle = MagicMock()
        player = ScanAgent(start_listening=False)
        mock_func = MagicMock()
        mock_a2m = MagicMock()
        mock_a2m.return_value = "order"
        mock_func.return_value = mock_a2m
        player.get_action_to_move_func = mock_func
        order = player.choose_move(mock_battle)
        assert order == "order"
        mock_input.assert_called_once()
        mock_func.assert_called_once()
        mock_a2m.assert_called_once_with(player, 1, mock_battle, None)


def test_choose_move_6():
    with patch("builtins.input") as mock_input:
        mock_input.return_value = 6
        mock_battle = MagicMock()
        player = ScanAgent(start_listening=False)
        mock_func = MagicMock()
        mock_a2m = MagicMock()
        mock_a2m.return_value = "order"
        mock_func.return_value = mock_a2m
        player.get_action_to_move_func = mock_func
        order = player.choose_move(mock_battle)
        assert order == "order"
        mock_input.assert_called_once()
        mock_func.assert_called_once()
        mock_a2m.assert_called_once_with(player, 6, mock_battle, None)
