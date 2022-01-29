# Manages the training cycle for RL agents
import asyncio
import copy
import datetime
import gc
import math
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns
import sys

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import set_start_method
from poke_env.player.baselines import SimpleHeuristicsPlayer, MaxBasePowerPlayer, RandomPlayer
from poke_env.player_configuration import _CONFIGURATION_FROM_PLAYER_COUNTER # noqa used for parallelism
from poke_env.server_configuration import LocalhostServerConfiguration
from poke_env.player.utils import evaluate_player
from progress.bar import IncrementalBar

from agents.basic_rl import SimpleRLAgent
from agents.expert_rl import ExpertRLAgent
from agents.sarsa_stark import SarsaStark, ExpertSarsaStark
from utils import InvalidArgument


async def main():
    current_time = datetime.datetime.now()
    current_time_string = current_time.strftime('%d-%m-%Y %H-%M-%S')
    os.makedirs('./logs', exist_ok=True)
    eval_challenges = 2000
    placement = 40
    agent_type = sys.argv[3].strip()
    if not sys.argv[1].isnumeric():
        raise InvalidArgument(f'{sys.argv[1]} should be an integer containing the number of battles for the training')
    challenges = int(sys.argv[1])
    if agent_type == 'simpleRL':
        path = f'./models/simpleRL/{sys.argv[2]}'
        agent = SimpleRLAgent(training=True, battle_format=sys.argv[2],
                              server_configuration=LocalhostServerConfiguration)
        update_agent = get_simple_rl
    elif agent_type == 'expertRL':
        path = f'./models/expertRL/{sys.argv[2]}'
        agent = ExpertRLAgent(training=True, battle_format=sys.argv[2],
                              server_configuration=LocalhostServerConfiguration)
        update_agent = get_expert_rl
    elif agent_type == 'SarsaStark':
        path = f'./models/SarsaStark/{sys.argv[2]}'
        agent = SarsaStark(training=True, battle_format=sys.argv[2],
                           server_configuration=LocalhostServerConfiguration)
        update_agent = get_sarsa_stark
    elif agent_type == 'expertSarsaStark':
        path = f'./models/expertSarsaStark/{sys.argv[2]}'
        agent = ExpertSarsaStark(training=True, battle_format=sys.argv[2],
                                 server_configuration=LocalhostServerConfiguration)
        update_agent = get_expert_sarsa_stark
    else:
        raise InvalidArgument(f'{agent_type} is not a valid RL agent')
    if path:
        os.makedirs(path, exist_ok=True)
    opponent1 = SimpleHeuristicsPlayer(server_configuration=LocalhostServerConfiguration)
    opponent2 = MaxBasePowerPlayer(server_configuration=LocalhostServerConfiguration)
    opponent3 = RandomPlayer(server_configuration=LocalhostServerConfiguration)
    evaluations = []
    cycles = []
    states = []
    bar = ProgressBar('Training', max=challenges * 3)
    max_group = challenges**(3 / 4)
    group = 1
    for j in range(math.ceil(max_group), 1, -1):
        if challenges % j == 0:
            group = j
            break
    for _ in range(challenges // group):
        pool = ProcessPoolExecutor()
        res = pool.submit(evaluate, update_agent, agent.get_model(), eval_challenges, placement,
                          _CONFIGURATION_FROM_PLAYER_COUNTER.copy())
        cycles.append(bar.index)
        states.append(len(agent.get_model()))
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
        evaluations.append(res.result())
        pool.shutdown(wait=True, cancel_futures=True)
        gc.collect()
    pool = ProcessPoolExecutor()
    res = pool.submit(evaluate, update_agent, agent.get_model(), eval_challenges, placement,
                      _CONFIGURATION_FROM_PLAYER_COUNTER.copy())
    cycles.append(bar.index)
    states.append(len(agent.get_model()))
    evaluations.append(res.result())
    pool.shutdown(wait=True, cancel_futures=True)
    bar.finish()
    sns.set_theme()
    sns.set_palette('colorblind')
    to_plot = []
    for evaluation in evaluations:
        to_plot.append(evaluation)
    plt.plot(cycles, to_plot)
    plt.savefig(f'./logs/training {current_time_string}.png', backend='agg', dpi=300)
    plt.clf()
    plt.plot(cycles, states)
    plt.savefig(f'./logs/state number {current_time_string}.png', backend='agg', dpi=300)
    with open(path + '/best.pokeai', 'wb') as file:
        pickle.dump(agent.get_model(), file)


def get_simple_rl(model, training=False, keep_training=False, max_concurrent_battles=1):
    model_copy = copy.deepcopy(model)
    return SimpleRLAgent(training=training, battle_format=sys.argv[2],
                         server_configuration=LocalhostServerConfiguration, model=model_copy,
                         keep_training=keep_training, max_concurrent_battles=max_concurrent_battles)


def get_expert_rl(model, training=False, keep_training=False, max_concurrent_battles=1):
    model_copy = copy.deepcopy(model)
    return ExpertRLAgent(training=training, battle_format=sys.argv[2],
                         server_configuration=LocalhostServerConfiguration, model=model_copy,
                         keep_training=keep_training, max_concurrent_battles=max_concurrent_battles)


def get_sarsa_stark(model, training=False, keep_training=False, max_concurrent_battles=1):
    model_copy = copy.deepcopy(model)
    return SarsaStark(training=training, battle_format=sys.argv[2],
                      server_configuration=LocalhostServerConfiguration, model=model_copy,
                      keep_training=keep_training, max_concurrent_battles=max_concurrent_battles)


def get_expert_sarsa_stark(model, training=False, keep_training=False, max_concurrent_battles=1):
    model_copy = copy.deepcopy(model)
    return ExpertSarsaStark(training=training, battle_format=sys.argv[2],
                            server_configuration=LocalhostServerConfiguration, model=model_copy,
                            keep_training=keep_training, max_concurrent_battles=max_concurrent_battles)


def evaluate(update_agent_func, model, challenges, placement, counter):
    from poke_env.player_configuration import _CONFIGURATION_FROM_PLAYER_COUNTER # noqa used for parallelism
    _CONFIGURATION_FROM_PLAYER_COUNTER.clear()
    _CONFIGURATION_FROM_PLAYER_COUNTER.update(counter)
    agent = update_agent_func(model, False, False, 10)
    evaluation = asyncio.get_event_loop().run_until_complete(evaluate_player(agent, challenges, placement))
    return evaluation[0]


class ProgressBar(IncrementalBar):
    width = 100
    suffix = IncrementalBar.suffix + ' ETA: %(eta_str)s'

    @property
    def eta_str(self):
        remaining_seconds = self.eta
        if remaining_seconds == 0:
            return '---'
        display_years = remaining_seconds // 946_080_000
        remaining_seconds = remaining_seconds - (display_years * 946_080_000)
        display_months = remaining_seconds // 2_592_000
        remaining_seconds = remaining_seconds - (display_months * 2_592_000)
        display_days = remaining_seconds // 86_400
        remaining_seconds = remaining_seconds - (display_days * 86_400)
        display_hours = remaining_seconds // 3600
        remaining_seconds = remaining_seconds - (display_hours * 3600)
        display_minutes = remaining_seconds // 60
        remaining_seconds = remaining_seconds - (display_minutes * 60)
        if remaining_seconds >= 60:
            raise RuntimeError('Error in computing ETA')
        display_seconds = remaining_seconds
        return_string = ''
        if display_years > 0:
            return_string += f'{display_years}y'
        if display_months > 0:
            return_string += f'{display_months}m'
        if display_days > 0:
            return_string += f'{display_days}d'
        if display_hours > 0 and not display_years > 0 and not display_months > 0:
            return_string += f'{display_hours}h'
        if display_minutes > 0 and not display_years > 0 and not display_months > 0:
            return_string += f'{display_minutes}m'
        if display_seconds > 0 and not display_years > 0 and not display_months > 0 and not display_days > 0:
            return_string += f'{display_seconds}s'
        return return_string


if __name__ == '__main__':
    set_start_method('spawn')
    asyncio.get_event_loop().run_until_complete(main())
