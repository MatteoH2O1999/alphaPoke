# Function to initialize the action space for RL custom algorithms
from typing import List


def init_action_space(battle_format) -> List[int]:
    action_space = []
    if battle_format == 'gen8randombattle':
        for _ in range(22):
            action_space.append(0)
    else:
        raise UnsupportedBattleFormat(f'{battle_format} is not a valid battle format')
    return action_space


class UnsupportedBattleFormat(Exception):
    pass
