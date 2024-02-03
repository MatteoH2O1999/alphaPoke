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
# Simple agent to analyze battle information
import asyncio

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.random_player import RandomPlayer
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
        return action_to_move_func(self, int(action), battle, None)

    def get_action_to_move_func(self):
        return get_int_action_to_move(self.format, self.format_is_doubles)


if __name__ == "__main__":  # pragma: no cover

    async def main(battle_format: str):
        opponent = RandomPlayer(battle_format=battle_format)
        player = ScanAgent(battle_format=battle_format)
        await player.battle_against(opponent, 1)

    asyncio.get_event_loop().run_until_complete(main(input("Insert battle format: ")))
