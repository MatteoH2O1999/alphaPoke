import asyncio
import pytest

from poke_env.player.baselines import RandomPlayer
from poke_env.server_configuration import LocalhostServerConfiguration
from unittest.mock import patch

from play import main


@pytest.mark.asyncio
@pytest.mark.timeout(180)
@pytest.mark.flaky
async def test_play_integration():
    opponent = RandomPlayer(battle_format="gen8randombattle")
    for _ in range(10):
        with patch("builtins.input") as mock_input, patch(
            "play.ShowdownServerConfiguration", LocalhostServerConfiguration
        ), patch("play.PlayerConfiguration") as mock_player, patch(
            "play.getpass"
        ) as mock_password:
            mock_input.side_effect = ["dad", "gen8randombattle", "", opponent.username]
            mock_player.return_value = None
            mock_password.return_value = ""
            await asyncio.gather(main(), opponent.accept_challenges(None, 1))
    assert opponent.n_finished_battles == 10
