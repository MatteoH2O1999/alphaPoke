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
import asyncio

from logging import disable, NOTSET, CRITICAL
from threading import Thread

from poke_env.player.player import Player


def close_player(player: Player):
    t = _ClosePlayerThread(player)
    t.start()
    t.join()


class _ClosePlayerThread(Thread):
    def __init__(self, player: Player, *args, **kwargs):
        self.player = player
        super().__init__(*args, **kwargs)

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.stop_player())

    async def stop_player(self):
        from agents.base_classes.tf_player import TFPlayer

        disable(CRITICAL)
        await self.player.stop_listening()
        if isinstance(self.player, TFPlayer):
            self.clean_tf_player()
        disable(NOTSET)

    def clean_tf_player(self):
        from agents.base_classes.tf_player import TFPlayer

        assert isinstance(self.player, TFPlayer)
        self.player.policy = None
