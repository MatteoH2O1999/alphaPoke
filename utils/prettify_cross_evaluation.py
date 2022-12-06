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
# Prettifies a poke-env cross evaluation dictionary
from tabulate import tabulate


def prettify_evaluation(evaluation_dict: dict):
    player_list = list(evaluation_dict.keys())
    if not __check_useful_numbers(evaluation_dict.keys()):
        tmp = {}
        for p_1, results in evaluation_dict.items():
            tmp_result = {}
            for p_2, win_rate in results.items():
                if not __check_useful_number(p_2, player_list):
                    tmp_result[__cut_player_number(p_2)] = win_rate
                else:
                    tmp_result[p_2] = win_rate
            if not __check_useful_number(p_1, player_list):
                tmp[__cut_player_number(p_1)] = tmp_result
            else:
                tmp[p_1] = tmp_result
        evaluation_dict = tmp
    players = []
    player_dict = {}
    for i, p in enumerate(evaluation_dict):
        players.append(p)
        player_dict[p] = i
    table = [["-"] + [p for p in evaluation_dict]]
    for p_1, results in evaluation_dict.items():
        to_append = [p_1]
        to_append.extend([None for _ in evaluation_dict])
        for p_2, win_rate in results.items():
            to_append[1 + player_dict[p_2]] = win_rate
        table.append(to_append)
    return tabulate(table)


def __check_useful_numbers(players) -> bool:
    player_list = []
    for p in players:
        player_list.append(__cut_player_number(p))
    for p in player_list:
        if not __check_useful_number(p, players):
            return False
    return True


def __check_useful_number(player, players) -> bool:
    player = __cut_player_number(player)
    player_list = []
    for p in players:
        player_list.append(__cut_player_number(p))
    count = 0
    for p in player_list:
        if player == p:
            count += 1
    if count > 1:
        return True
    return False


def __cut_player_number(player):
    player_split = player.split(" ")
    if player_split[-1].isnumeric() and len(player_split) > 1:
        player_split = player_split[:-1]
    return " ".join(player_split)
