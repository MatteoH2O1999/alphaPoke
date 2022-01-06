import asyncio
import matplotlib.pyplot as plt
import math
import sys

from poke_env.player.utils import evaluate_player
from poke_env.server_configuration import LocalhostServerConfiguration
from tabulate import tabulate

from cross_eval import InvalidArgument
from utils import create_agent
from utils.plot_eval import plot_eval
from utils.prettify_cross_evaluation import __cut_player_number


async def main():
    if not sys.argv[1].isnumeric():
        raise InvalidArgument(f'{sys.argv[1]} is not the correct format for the number of challenges to evaluate.')
    challenges = int(sys.argv[1])
    battle_format = sys.argv[2]
    players = []
    used_players = []
    for i in range(3, len(sys.argv)):
        agent_name = sys.argv[i]
        if agent_name not in used_players:
            used_players.append(agent_name)
            to_append = create_agent(agent_name, None, battle_format, False, LocalhostServerConfiguration,
                                     False, 50)
            for p in to_append:
                players.append(p)
    results = [['Player', 'Evaluation']]
    for player in players:
        evaluation = await evaluate_player(player, challenges, math.ceil(math.log2(challenges) * 2))
        results.append([__cut_player_number(player.username), evaluation])
    print(tabulate(results))
    plt.figure(dpi=600)
    plot_eval(results, False, './evaluation plots')


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
