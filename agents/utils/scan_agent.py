# Simple agent to analyze battle information
import asyncio

from functools import lru_cache
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.baselines import RandomPlayer
from poke_env.player.player import Player
from typing import Awaitable, Union

from utils.action_to_move_function import get_int_action_to_move


class ScanAgent(Player):
    def choose_move(
        self, battle: AbstractBattle
    ) -> Union[BattleOrder, Awaitable[BattleOrder]]:
        print("Insert action int.")
        action = input("Action: ")
        action_to_move_func = self.get_action_to_move_func()
        return action_to_move_func(self, int(action), battle, False)

    @lru_cache(1)
    def get_action_to_move_func(self):
        return get_int_action_to_move(self.format, self.format_is_doubles)


async def main(battle_format: str):
    opponent = RandomPlayer(battle_format=battle_format)
    player = ScanAgent(battle_format=battle_format)
    await player.battle_against(opponent, 1)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main(input("Insert battle format: ")))
