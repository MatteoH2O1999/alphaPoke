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
# Play against a specified username
#####################################################################
# Usage: run the script and follow the instructions of the terminal #
#####################################################################
import asyncio

from getpass import getpass
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration

from utils.create_agent import create_agent


async def main():
    ai_type = input("Bot type: ")
    battle_format = input("Battle format: ")
    ai_username = input("Bot username: ")
    ai_password = getpass("Bot password: ")
    player_username = input("Play against: ")
    player_conf = PlayerConfiguration(ai_username, ai_password)
    agent = create_agent(
        ai_type,
        battle_format,
        player_conf,
        ShowdownServerConfiguration,
        False,
        False,
        1,
    )[0]
    await agent.send_challenges(player_username, 1)


if __name__ == "__main__":  # pragma: no cover
    asyncio.get_event_loop().run_until_complete(main())
