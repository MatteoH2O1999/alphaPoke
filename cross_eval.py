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
####################################################################################
# Usage: python cross_eval.py NUM_CHALLENGES BATTLE_FORMAT [(AGENT_TYPE QUANTITY)] #
#                                                                                  #
# Example: python cross_eval.py 1000 gen8randombattle dad 1 expertRL-best 2        #
####################################################################################
import asyncio
import sys

from poke_env.server_configuration import LocalhostServerConfiguration
from poke_env.player.player import Player
from poke_env.player.utils import cross_evaluate
from typing import List
from utils import InvalidArgument, InvalidArgumentNumber
from utils.create_agent import create_agent
from utils.prettify_cross_evaluation import prettify_evaluation


async def main():
    if len(sys.argv) % 2 == 0:
        raise InvalidArgumentNumber(
            f"Wrong number of arguments. Correct format:\n"
            f"NUMBER_OF_CHALLENGES BATTLE_FORMAT [(AGENT_NAME, NUMBER_OF_AGENTS)]+\n"
        )
    players = []
    challenges = 0
    battle_format = ""
    for i in range(1, len(sys.argv)):
        if i == 1:
            if not sys.argv[i].isnumeric():
                raise InvalidArgument(
                    f"{sys.argv[i]} is not in the correct format. It should be an integer number to"
                    f"specify how many challenges each pair of agents needs to play.\n"
                )
            challenges = int(sys.argv[i])
            continue
        if i == 2:
            battle_format = sys.argv[i]
            continue
        if i % 2 == 1:
            if not sys.argv[i + 1].isnumeric():
                raise InvalidArgument(
                    f"{sys.argv[i + 1]} is not in the correct format. It should be an integer"
                    f"number to specify how many agents of type {sys.argv[i]} to put in the"
                    f"evaluation process.\n"
                )
            agent_name = sys.argv[i]
            agent_quantity = int(sys.argv[i + 1])
            for _ in range(agent_quantity):
                to_append = create_agent(
                    agent_name,
                    battle_format,
                    None,
                    LocalhostServerConfiguration,
                    False,
                    False,
                    30,
                )
                for p in to_append:
                    players.append(p)
    await cross_evaluate_players(players, challenges)


async def cross_evaluate_players(players: List[Player], challenges: int):
    evaluation = await cross_evaluate(players, n_challenges=challenges)
    print(prettify_evaluation(evaluation))


if __name__ == "__main__":  # pragma: no cover
    asyncio.get_event_loop().run_until_complete(main())
