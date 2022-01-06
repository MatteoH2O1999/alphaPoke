import asyncio
import sys

from poke_env.server_configuration import LocalhostServerConfiguration
from poke_env.player.utils import cross_evaluate
from utils import create_agent
from utils.prettify_cross_evaluation import prettify_evaluation


async def main():
    if len(sys.argv) % 2 == 0:
        raise InvalidArgumentNumber(f'Wrong number of arguments. Correct format:\n'
                                    f'NUMBER_OF_CHALLENGES BATTLE_FORMAT [(AGENT_NAME, NUMBER_OF_AGENTS)]+\n')
    players = []
    challenges = 0
    battle_format = ''
    for i in range(1, len(sys.argv)):
        if i == 1:
            if not sys.argv[i].isnumeric():
                raise InvalidArgument(f'{sys.argv[i]} is not in the correct format. It should be an integer number to'
                                      f'specify how many challenges each pair of agents needs to play.\n')
            challenges = int(sys.argv[i])
            continue
        if i == 2:
            battle_format = sys.argv[i]
            continue
        if i % 2 == 1:
            if not sys.argv[i + 1].isnumeric():
                raise InvalidArgument(f'{sys.argv[i + 1]} is not in the correct format. It should be an integer'
                                      f'number to specify how many agents of type {sys.argv[i]} to put in the'
                                      f'evaluation process.\n')
            agent_name = sys.argv[i]
            agent_quantity = int(sys.argv[i + 1])
            for _ in range(agent_quantity):
                to_append = create_agent(agent_name, None, battle_format, False, LocalhostServerConfiguration,
                                         False, 30)
                for p in to_append:
                    players.append(p)
    evaluation = await cross_evaluate(players, n_challenges=challenges)
    print(prettify_evaluation(evaluation))


class InvalidArgumentNumber(Exception):
    pass


class InvalidArgument(Exception):
    pass


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
