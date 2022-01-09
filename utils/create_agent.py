# Function to parse cli strings into agents
import os
import pickle

from poke_env.player.player import Player
from typing import List

from agents.dad import Dad
from agents.eight_year_old_me import EightYearOldMe
from agents.twenty_year_old_me import TwentyYearOldMe
from agents.basic_rl import SimpleRLAgent


def create_agent(cli_name, player_configuration, battle_format, start_timer, server_configuration,
                 save_replay=False, concurrent=1, **others) -> List[Player]:
    agent_name = cli_name.strip()
    kwargs = dict(
        player_configuration=player_configuration,
        battle_format=battle_format,
        max_concurrent_battles=concurrent,
        save_replays=save_replay,
        start_timer_on_battle_start=start_timer,
        server_configuration=server_configuration
    )
    for key, value in others.items():
        kwargs[key] = value
    if agent_name == 'dad':
        agent = [Dad(**kwargs)]
    elif agent_name == '8-year-old-me':
        agent = [EightYearOldMe(**kwargs)]
    elif agent_name == '20-year-old-me':
        agent = [TwentyYearOldMe(**kwargs)]
    elif 'simpleRL-best' in agent_name:
        with open('./models/simpleRL/best.pokeai', 'b') as model_file:
            model = pickle.load(model_file)
        keep_training = False
        if 'train' in agent_name:
            keep_training = True
        agent = [SimpleRLAgent(**kwargs, keep_training=keep_training, model=model)]
    elif 'simpleRL-all' in agent_name:
        models = []
        files = os.listdir('./models/simpleRL')
        for file in files:
            with open('./models/simpleRL/' + file, 'b') as model_file:
                models.append(pickle.load(model_file))
        agent = []
        keep_training = False
        if 'train' in agent_name:
            keep_training = True
        for model in models:
            agent.append(SimpleRLAgent(**kwargs, keep_training=keep_training, model=model))
    else:
        raise UnsupportedAgentType(f'{cli_name} is not a valid agent type')
    return agent


class UnsupportedAgentType(Exception):
    pass
