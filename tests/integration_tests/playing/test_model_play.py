import asyncio
import pytest

from poke_env.player.random_player import RandomPlayer

from utils.close_player import close_player
from utils.create_agent import create_agent

from conftest import agents

AGENTS = agents()


@pytest.mark.parametrize("cli_name,expected_class,battle_format", AGENTS)
@pytest.mark.asyncio
async def test_single_battle_opponent(cli_name, expected_class, battle_format):
    opponent = RandomPlayer(battle_format=battle_format)
    player = create_agent(cli_name, battle_format=battle_format)[0]
    assert isinstance(player, expected_class)
    await opponent.battle_against(player, 1)
    close_player(player)
    close_player(opponent)


@pytest.mark.parametrize("cli_name,expected_class,battle_format", AGENTS)
@pytest.mark.asyncio
async def test_single_battle_player(cli_name, expected_class, battle_format):
    opponent = RandomPlayer(battle_format=battle_format)
    player = create_agent(cli_name, battle_format=battle_format)[0]
    assert isinstance(player, expected_class)
    await player.battle_against(opponent, 1)
    close_player(player)
    close_player(opponent)


@pytest.mark.parametrize("cli_name,expected_class,battle_format", AGENTS)
@pytest.mark.asyncio
async def test_multiple_battle_opponent(cli_name, expected_class, battle_format):
    opponent = RandomPlayer(battle_format=battle_format)
    player = create_agent(cli_name, battle_format=battle_format)[0]
    assert isinstance(player, expected_class)
    await opponent.battle_against(player, 2)
    await opponent.battle_against(player, 3)
    close_player(player)
    close_player(opponent)


@pytest.mark.parametrize("cli_name,expected_class,battle_format", AGENTS)
@pytest.mark.asyncio
async def test_multiple_battle_player(cli_name, expected_class, battle_format):
    opponent = RandomPlayer(battle_format=battle_format)
    player = create_agent(cli_name, battle_format=battle_format)[0]
    assert isinstance(player, expected_class)
    await player.battle_against(opponent, 2)
    await player.battle_against(opponent, 3)
    close_player(player)
    close_player(opponent)


@pytest.mark.parametrize("cli_name,expected_class,battle_format", AGENTS)
@pytest.mark.asyncio
async def test_single_accept(cli_name, expected_class, battle_format):
    opponent = RandomPlayer(battle_format=battle_format)
    player = create_agent(cli_name, battle_format=battle_format)[0]
    assert isinstance(player, expected_class)
    await asyncio.gather(
        player.accept_challenges(opponent.username, 1),
        opponent.send_challenges(player.username, 1, to_wait=player.logged_in),
    )
    close_player(player)
    close_player(opponent)


@pytest.mark.parametrize("cli_name,expected_class,battle_format", AGENTS)
@pytest.mark.asyncio
async def test_multiple_accept(cli_name, expected_class, battle_format):
    opponent = RandomPlayer(battle_format=battle_format)
    player = create_agent(cli_name, battle_format=battle_format)[0]
    assert isinstance(player, expected_class)
    await asyncio.gather(
        player.accept_challenges(opponent.username, 2),
        opponent.send_challenges(player.username, 2, to_wait=player.logged_in),
    )
    await asyncio.gather(
        player.accept_challenges(opponent.username, 3),
        opponent.send_challenges(player.username, 3, to_wait=player.logged_in),
    )
    close_player(player)
    close_player(opponent)


@pytest.mark.parametrize("cli_name,expected_class,battle_format", AGENTS)
@pytest.mark.asyncio
async def test_single_send(cli_name, expected_class, battle_format):
    opponent = RandomPlayer(battle_format=battle_format)
    player = create_agent(cli_name, battle_format=battle_format)[0]
    assert isinstance(player, expected_class)
    await asyncio.gather(
        player.send_challenges(opponent.username, 1, to_wait=opponent.logged_in),
        opponent.accept_challenges(player.username, 1),
    )
    close_player(player)
    close_player(opponent)


@pytest.mark.parametrize("cli_name,expected_class,battle_format", AGENTS)
@pytest.mark.asyncio
async def test_multiple_send(cli_name, expected_class, battle_format):
    opponent = RandomPlayer(battle_format=battle_format)
    player = create_agent(cli_name, battle_format=battle_format)[0]
    assert isinstance(player, expected_class)
    await asyncio.gather(
        player.send_challenges(opponent.username, 2, to_wait=opponent.logged_in),
        opponent.accept_challenges(player.username, 2),
    )
    await asyncio.gather(
        player.send_challenges(opponent.username, 3, to_wait=opponent.logged_in),
        opponent.accept_challenges(player.username, 3),
    )
    close_player(player)
    close_player(opponent)
