import asyncio
import pytest

from poke_env.player.baselines import RandomPlayer
from poke_env.server_configuration import LocalhostServerConfiguration
from unittest.mock import patch

from play import main


@pytest.mark.asyncio
@pytest.mark.timeout(90)
@pytest.mark.flaky(max_runs=10, min_passes=1)
async def test_play_integration():
    opponent = RandomPlayer(battle_format="gen8randombattle")
    opponent_task = asyncio.ensure_future(opponent.accept_challenges(None, 10))
    for _ in range(10):
        with patch("builtins.input") as mock_input, patch(
            "play.ShowdownServerConfiguration", LocalhostServerConfiguration
        ), patch("play.PlayerConfiguration") as mock_player, patch(
            "play.getpass"
        ) as mock_password:
            mock_input.side_effect = ["dad", "gen8randombattle", "", opponent.username]
            mock_player.return_value = None
            mock_password.return_value = ""
            await main()
    await opponent_task
    assert opponent.n_finished_battles == 10
