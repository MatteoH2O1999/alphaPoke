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
###############################################################
# Usage: python eval.py NUM_CHALLENGES SAVE_PLOT [AGENT_TYPE] #
#                                                             #
# Example: python eval.py 10000 False dad expertRL-best       #
###############################################################
import asyncio
import matplotlib.pyplot as plt
import sys

from poke_env.player.player import Player
from poke_env.player.utils import evaluate_player
from poke_env.server_configuration import LocalhostServerConfiguration
from tabulate import tabulate
from typing import Iterable

from utils import InvalidArgument
from utils.create_agent import create_agent
from utils.plot_eval import plot_eval
from utils.prettify_cross_evaluation import __cut_player_number


async def main():
    if not sys.argv[1].isnumeric():
        raise InvalidArgument(
            f"{sys.argv[1]} is not the correct format for the number of challenges to evaluate."
        )
    challenges = int(sys.argv[1])
    battle_format = "gen8randombattle"
    if sys.argv[2].lower() in ["true", "t", "y", "yes"]:
        save = True
    elif sys.argv[2].lower() in ["false", "f", "n", "no"]:
        save = False
    else:
        raise InvalidArgument(f"{sys.argv[2]} is not a valid boolean argument.")
    players = []
    used_players = []
    for i in range(3, len(sys.argv)):
        agent_name = sys.argv[i]
        if agent_name not in used_players:
            used_players.append(agent_name)
            to_append = create_agent(
                agent_name,
                battle_format,
                None,
                LocalhostServerConfiguration,
                False,
                False,
                10,
            )
            for p in to_append:
                players.append(p)
    await evaluate_players(players, challenges, save)


async def evaluate_players(
    players: Iterable[Player], challenges: int, save: bool, save_path="./logs"
):
    results = [["Player", "Evaluation"]]
    for player in players:
        evaluation = await evaluate_player(player, challenges, 40)
        results.append([__cut_player_number(player.username), evaluation])
    print(tabulate(results))
    if save:
        plt.figure(dpi=600)
    else:
        plt.figure()
    plot_eval(results, save, save_path)


if __name__ == "__main__":  # pragma: no cover
    asyncio.get_event_loop().run_until_complete(main())
