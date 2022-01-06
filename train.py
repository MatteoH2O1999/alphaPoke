# Manages the training cycle for RL agents
import asyncio
import copy
import gc
import math
import pickle
import os
import sys

from cross_eval import InvalidArgument
from agents.basic_rl import SimpleRLAgent
from poke_env.player.baselines import SimpleHeuristicsPlayer, MaxBasePowerPlayer, RandomPlayer
from poke_env.server_configuration import LocalhostServerConfiguration
from poke_env.player.utils import evaluate_player
from progress.bar import IncrementalBar


async def main():
    agent_type = sys.argv[3].strip()
    update_agent = default_update_agent
    if not sys.argv[1].isnumeric():
        raise InvalidArgument(f'{sys.argv[1]} should be an integer containing the number of battles for the training')
    challenges = int(sys.argv[1])
    if agent_type == 'simpleRL':
        path = './models/simpleRL'
        os.makedirs(path, exist_ok=True)
        agent = SimpleRLAgent(training=True, battle_format=sys.argv[2],
                              server_configuration=LocalhostServerConfiguration)
        update_agent = get_simple_rl
    else:
        raise InvalidArgument(f'{agent_type} is not a valid RL agent')
    opponent1 = SimpleHeuristicsPlayer(server_configuration=LocalhostServerConfiguration)
    opponent2 = MaxBasePowerPlayer(server_configuration=LocalhostServerConfiguration)
    opponent3 = RandomPlayer(server_configuration=LocalhostServerConfiguration)
    bar = IncrementalBar('Training', max=challenges * 3)
    bar.width = 100
    max_group = min(math.sqrt(challenges), 10000)
    group = 1
    for j in range(math.ceil(max_group), 1, -1):
        if challenges % j == 0:
            group = j
            break
    for _ in range(challenges // group):
        for _ in range(group):
            await agent.battle_against(opponent3, 1)
            bar.next()
            await agent.battle_against(opponent2, 1)
            bar.next()
            await agent.battle_against(opponent1, 1)
            bar.next()
        opponent1.reset_battles()
        opponent2.reset_battles()
        opponent3.reset_battles()
        agent.reset_battles()
        gc.collect()
    bar.finish()
    with open(path + '/best.pokeai', 'wb') as file:
        pickle.dump(agent.get_model(), file)
    eval_agent = update_agent(model=agent.get_model(), training=False, keep_training=False, max_concurrent_battles=10)
    if eval_agent:
        print(await evaluate_player(eval_agent, 2000, 40))
    else:
        print(await evaluate_player(agent, 2000, 40))


def default_update_agent(model, training=False, keep_training=False, max_concurrent_battle=1):
    return None


def get_simple_rl(model, training=False, keep_training=False, max_concurrent_battles=1):
    model_copy = copy.deepcopy(model)
    return SimpleRLAgent(training=training, battle_format=sys.argv[2],
                         server_configuration=LocalhostServerConfiguration, model=model_copy,
                         keep_training=keep_training, max_concurrent_battles=max_concurrent_battles)


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
