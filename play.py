# Play against a specified username
import asyncio

from getpass import getpass
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration

from utils.create_agent import create_agent


async def main():
    ai_type = input('Bot type: ')
    battle_format = input('Battle format: ')
    ai_username = input('Bot username: ')
    ai_password = getpass('Bot password: ')
    player_username = input('Play against: ')
    player_conf = PlayerConfiguration(ai_username, ai_password)
    agent = create_agent(ai_type, player_conf, battle_format, False, ShowdownServerConfiguration, True, 1)[0]
    await agent.send_challenges(player_username, 1)


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
