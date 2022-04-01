import pytest

from poke_env.environment.battle import Battle
from poke_env.player.battle_order import ForfeitBattleOrder
from unittest.mock import MagicMock

from agents.base_classes.tf_player import TFPlayer, _Env


def test_env():
    username = "Username"
    calc_reward = MagicMock()
    calc_reward.return_value = 69.0
    action_to_move = MagicMock()
    action_to_move.return_value = ForfeitBattleOrder()
    embed_battle = MagicMock()
    embed_battle.return_value = [0, 1, 2]
    embedding_description = MagicMock()
    action_space_size = 42
    opponents = ["Opponent 1", "Opponent 2"]
    env = _Env(
        username,
        calc_reward,  # noqa
        action_to_move,  # noqa
        embed_battle,
        embedding_description,
        action_space_size,
        opponents,
        start_listening=False,
        start_challenging=False,
    )
    current_battle = Battle("tag", "username", None)  # noqa
    last_battle = Battle("tag", "username", None)  # noqa
    assert env.username == "Username 1"
    assert env.calc_reward(last_battle, current_battle) == 69.0
    assert isinstance(env.action_to_move(3, current_battle), ForfeitBattleOrder)
    assert env.embed_battle(current_battle) == [0, 1, 2]
    assert env.describe_embedding() is embedding_description
    assert env.action_space_size() == 42
    assert env.get_opponent() == ["Opponent 1", "Opponent 2"]
    calc_reward.assert_called_once_with(last_battle, current_battle)
    action_to_move.assert_called_once_with(env.agent, 3, current_battle)
    embed_battle.assert_called_once_with(current_battle)
    embedding_description.assert_not_called()
