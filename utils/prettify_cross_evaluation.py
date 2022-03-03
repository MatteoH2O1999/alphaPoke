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
    table = [["-"] + [p for p in evaluation_dict]]
    for p_1, results in evaluation_dict.items():
        to_append = [p_1]
        for p_2, win_rate in results.items():
            to_append.append(evaluation_dict[p_1][p_2])
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
    if player_split[-1].isnumeric():
        player_split = player_split[:-1]
    return " ".join(player_split)
