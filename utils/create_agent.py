# Function to parse cli strings into agents
import os
import pickle

from poke_env.player.player import Player
from poke_env.server_configuration import LocalhostServerConfiguration
from typing import List

from agents.dad import Dad
from agents.eight_year_old_me import EightYearOldMe
from agents.twenty_year_old_me import TwentyYearOldMe
from agents.basic_rl import SimpleRLAgent
from agents.expert_rl import ExpertRLAgent
from agents.sarsa_stark import SarsaStark, ExpertSarsaStark


def create_agent(
    cli_name,
    battle_format,
    player_configuration=None,
    server_configuration=LocalhostServerConfiguration,
    start_timer=False,
    save_replay=False,
    max_concurrent_battles=1,
    **others,
) -> List[Player]:
    agent_name = cli_name.strip()
    kwargs = dict(
        player_configuration=player_configuration,
        battle_format=battle_format,
        max_concurrent_battles=max_concurrent_battles,
        save_replays=save_replay,
        start_timer_on_battle_start=start_timer,
        server_configuration=server_configuration,
    )
    for key, value in others.items():
        kwargs[key] = value
    if agent_name == "dad":
        agent = [Dad(**kwargs)]
    elif agent_name == "8-year-old-me":
        agent = [EightYearOldMe(**kwargs)]
    elif agent_name == "20-year-old-me":
        agent = [TwentyYearOldMe(**kwargs)]
    elif "simpleRL-best" in agent_name:
        with open(f"./models/simpleRL/{battle_format}/best.pokeai", "rb") as model_file:
            model = pickle.load(model_file)
        keep_training = False
        if "train" in agent_name:
            keep_training = True
        agent = [SimpleRLAgent(**kwargs, keep_training=keep_training, model=model)]
    elif "simpleRL-all" in agent_name:
        models = []
        files = os.listdir(f"./models/simpleRL/{battle_format}")
        for file in files:
            with open(f"./models/simpleRL/{battle_format}/" + file, "rb") as model_file:
                models.append(pickle.load(model_file))
        agent = []
        keep_training = False
        if "train" in agent_name:
            keep_training = True
        for model in models:
            agent.append(
                SimpleRLAgent(**kwargs, keep_training=keep_training, model=model)
            )
    elif "expertRL-best" in agent_name:
        with open(f"./models/expertRL/{battle_format}/best.pokeai", "rb") as model_file:
            model = pickle.load(model_file)
        keep_training = False
        if "train" in agent_name:
            keep_training = True
        agent = [ExpertRLAgent(**kwargs, keep_training=keep_training, model=model)]
    elif "expertRL-all" in agent_name:
        models = []
        files = os.listdir(f"./models/expertRL/{battle_format}")
        for file in files:
            with open(f"./models/expertRL/{battle_format}/" + file, "rb") as model_file:
                models.append(pickle.load(model_file))
        agent = []
        keep_training = False
        if "train" in agent_name:
            keep_training = True
        for model in models:
            agent.append(
                ExpertRLAgent(**kwargs, keep_training=keep_training, model=model)
            )
    elif "simpleSarsaStark-best" in agent_name:
        with open(
            f"./models/SarsaStark/{battle_format}/best.pokeai", "rb"
        ) as model_file:
            model = pickle.load(model_file)
        keep_training = False
        if "train" in agent_name:
            keep_training = True
        agent = [SarsaStark(**kwargs, keep_training=keep_training, model=model)]
    elif "simpleSarsaStark-all" in agent_name:
        models = []
        files = os.listdir(f"./models/SarsaStark/{battle_format}")
        for file in files:
            with open(
                f"./models/SarsaStark/{battle_format}/" + file, "rb"
            ) as model_file:
                models.append(pickle.load(model_file))
        agent = []
        keep_training = False
        if "train" in agent_name:
            keep_training = True
        for model in models:
            agent.append(SarsaStark(**kwargs, keep_training=keep_training, model=model))
    elif "expertSarsaStark-best" in agent_name:
        with open(
            f"./models/expertSarsaStark/{battle_format}/best.pokeai", "rb"
        ) as model_file:
            model = pickle.load(model_file)
        keep_training = False
        if "train" in agent_name:
            keep_training = True
        agent = [ExpertSarsaStark(**kwargs, keep_training=keep_training, model=model)]
    elif "expertSarsaStark-all" in agent_name:
        models = []
        files = os.listdir(f"./models/expertSarsaStark/{battle_format}")
        for file in files:
            with open(
                f"./models/expertSarsaStark/{battle_format}/" + file, "rb"
            ) as model_file:
                models.append(pickle.load(model_file))
        agent = []
        keep_training = False
        if "train" in agent_name:
            keep_training = True
        for model in models:
            agent.append(
                ExpertSarsaStark(**kwargs, keep_training=keep_training, model=model)
            )
    else:
        raise UnsupportedAgentType(f"{cli_name} is not a valid agent type")
    return agent


class UnsupportedAgentType(Exception):
    pass
