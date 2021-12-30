# Function to parse cli strings into agents
from poke_env.player.player import Player
from agents.dad import Dad
from agents.eight_year_old_me import EightYearOldMe
from agents.twenty_year_old_me import TwentyYearOldMe


def create_agent(cli_name, player_configuration, battle_format, start_timer, server_configuration,
                 save_replay: bool, concurrent=1) -> Player:
    agent_name = cli_name.strip()
    kwargs = dict(
        player_configuration=player_configuration,
        battle_format=battle_format,
        max_concurrent_battles=concurrent,
        save_replays=save_replay,
        start_timer_on_battle_start=start_timer,
        server_configuration=server_configuration
    )
    if agent_name == 'dad':
        agent = Dad(**kwargs)
    elif agent_name == '8-year-old-me':
        agent = EightYearOldMe(**kwargs)
    elif agent_name == '20-year-old-me':
        agent = TwentyYearOldMe(**kwargs)
    else:
        raise UnsupportedAgentType(f'{cli_name} is not a valid agent type')
    return agent


class UnsupportedAgentType(Exception):
    pass
