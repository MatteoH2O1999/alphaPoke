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
from utils.prettify_cross_evaluation import (
    prettify_evaluation,
    __cut_player_number,
    __check_useful_number,
    __check_useful_numbers,
)


def test_cut_player_number():
    player = "Player1"
    assert __cut_player_number(player) == "Player1"
    player = "Player 1"
    assert __cut_player_number(player) == "Player"
    player = "Player1 2"
    assert __cut_player_number(player) == "Player1"
    player = "123 456"
    assert __cut_player_number(player) == "123"
    player = "123"
    assert __cut_player_number(player) == "123"


def test_check_useful_number():
    players = ["Player 1", "Good Player 1"]
    assert not __check_useful_number(players[0], players)
    players.append("Player 2")
    assert not __check_useful_number(players[1], players)
    assert __check_useful_number(players[0], players)


def test_check_useful_numbers():
    players = ["Player 1", "Good Player 1"]
    assert not __check_useful_numbers(players)
    players.append("Player 2")
    assert not __check_useful_numbers(players)
    players = [f"Player {i}" for i in range(5)]
    assert __check_useful_numbers(players)


def test_prettify_cross_evaluation_without_number_cut():
    evaluations = {"p1": {"p2": 0.5}, "p2": {"p1": 0.5}}
    expected = "--  ---  ---\n-   p1   p2\np1       0.5\np2  0.5\n--  ---  ---"
    assert prettify_evaluation(evaluations) == expected
    evaluations = {"p1": {"p2": 0.5, "p1": None}, "p2": {"p1": 0.5, "p2": None}}
    assert prettify_evaluation(evaluations) == expected


def test_prettify_cross_evaluation_with_number_cut():
    evaluations = {"p1 1": {"p2 1": 0.5}, "p2 1": {"p1 1": 0.5}}
    expected = "--  ---  ---\n-   p1   p2\np1       0.5\np2  0.5\n--  ---  ---"
    assert prettify_evaluation(evaluations) == expected
    evaluations = {
        "p1 1": {"p2 1": 0.5, "p1 1": None},
        "p2 1": {"p1 1": 0.5, "p2 1": None},
    }
    assert prettify_evaluation(evaluations) == expected


def test_prettify_cross_evaluation_with_mixed_numbers():
    evaluations = {
        "p1 1": {"p2 1": 0.5, "p1 2": 0.5},
        "p2 1": {"p1 1": 0.5, "p1 2": 0.5},
        "p1 2": {"p1 1": 0.5, "p2 1": 0.5},
    }
    expected = (
        "----  ----  ---  ----\n"
        "-     p1 1  p2   p1 2\n"
        "p1 1        0.5  0.5\n"
        "p2    0.5        0.5\n"
        "p1 2  0.5   0.5\n"
        "----  ----  ---  ----"
    )
    assert prettify_evaluation(evaluations) == expected
    evaluations = {
        "p1 1": {"p2 1": 0.5, "p1 2": 0.5, "p1 1": None},
        "p2 1": {"p1 1": 0.5, "p1 2": 0.5, "p2 1": None},
        "p1 2": {"p1 1": 0.5, "p2 1": 0.5, "p1 2": None},
    }
    assert prettify_evaluation(evaluations) == expected
