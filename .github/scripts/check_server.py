import asyncio
from typing import Awaitable, Union

from poke_env.ps_client import LocalhostServerConfiguration
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import ForfeitBattleOrder, BattleOrder
from poke_env.player.random_player import RandomPlayer
from poke_env.player.player import Player

from utils.close_player import close_player


class ResetPlayer(Player):
    def choose_move(
        self, battle: AbstractBattle
    ) -> Union[BattleOrder, Awaitable[BattleOrder]]:
        return ForfeitBattleOrder()


async def main():
    print("Instantiating players...")
    player = ResetPlayer(
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen8randombattle",
    )
    opponent = RandomPlayer(
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen8randombattle",
    )
    print("Trying battle...")
    await player.battle_against(opponent, 1)
    print("Stopping players...")
    close_player(player)
    close_player(opponent)


if __name__ == "__main__":
    print("Waiting for showdown server...")
    redo = True
    while redo:
        redo = False
        try:
            print("Trying test battle...")
            asyncio.new_event_loop().run_until_complete(asyncio.wait_for(main(), 10))
            print("Test battle successful!!!")
        except asyncio.TimeoutError:
            print("Test battle timed out...")
            redo = True
            print("Retrying...")
