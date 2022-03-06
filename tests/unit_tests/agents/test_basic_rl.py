import pytest

from poke_env.environment.battle import Battle
from poke_env.environment.pokemon import Pokemon
from typing import List
from unittest.mock import patch, MagicMock

from agents import MON_HP_REWARD, MON_FAINTED_REWARD, VICTORY_REWARD
from agents.basic_rl import SimpleRLAgent, _battle_to_state_gen8random
from utils.action_to_move_function import action_to_move_gen8single
from utils.invalid_argument import InvalidArgument


def test_get_battle_to_state_func_failure():
    with patch("agents.basic_rl.SimpleRLAgent._get_action_to_move_func"), patch(
        "agents.basic_rl.SimpleRLAgent._get_action_space_size"
    ):
        with pytest.raises(InvalidArgument):
            SimpleRLAgent(start_listening=False, battle_format="nobattleformat")


def test_get_battle_to_state_func_success():
    agent = SimpleRLAgent(start_listening=False, battle_format="gen8randombattle")
    assert agent._get_battle_to_state_func() is _battle_to_state_gen8random


def test_get_action_to_move_func_failure():
    with patch("agents.basic_rl.SimpleRLAgent._get_battle_to_state_func"), patch(
        "agents.basic_rl.SimpleRLAgent._get_action_space_size"
    ):
        with pytest.raises(InvalidArgument):
            SimpleRLAgent(start_listening=False, battle_format="nobattleformat")


def test_get_action_to_move_func_success():
    agent = SimpleRLAgent(start_listening=False, battle_format="gen8randombattle")
    assert agent._get_action_to_move_func() is action_to_move_gen8single


def test_get_action_space_size_failure():
    with patch("agents.basic_rl.SimpleRLAgent._get_battle_to_state_func"), patch(
        "agents.basic_rl.SimpleRLAgent._get_action_to_move_func"
    ):
        with pytest.raises(InvalidArgument):
            SimpleRLAgent(start_listening=False, battle_format="nobattleformat")


def test_get_action_space_size_success():
    agent = SimpleRLAgent(start_listening=False, battle_format="gen8randombattle")
    assert agent._get_action_space_size() == 22


def test_train():
    model = {(1, 0): [[1, 0], 3, [2, 1]], (0, 1): [[0, 1], 5, [3, 2]]}
    agent = SimpleRLAgent(
        model=model, start_listening=False, battle_format="gen8randombattle"
    )
    agent._get_learning_rate = MagicMock()
    agent._get_learning_rate.return_value = 1.0
    agent._train((1, 0), 1, 0)
    agent._get_learning_rate.assert_called_once()
    assert agent.model == {(1, 0): [[1, 0], 4, [2, 2]], (0, 1): [[0, 1], 5, [3, 2]]}


def test_calc_reward_won():
    current_battle = Battle("battle_tag", "username", None)  # noqa
    last_battle = Battle("battle_tag", "username", None)  # noqa
    current_battle._won = True
    assert SimpleRLAgent._calc_reward(last_battle, current_battle) == VICTORY_REWARD


def test_calc_reward_lost():
    current_battle = Battle("battle_tag", "username", None)  # noqa
    last_battle = Battle("battle_tag", "username", None)  # noqa
    current_battle._won = False
    assert SimpleRLAgent._calc_reward(last_battle, current_battle) == -VICTORY_REWARD


def test_calc_reward_player():
    current_battle = Battle("battle_tag", "username", None)  # noqa
    last_battle = Battle("battle_tag", "username", None)  # noqa
    current_battle._team = {"charizard": Pokemon(species="charizard")}
    current_battle._team["charizard"]._max_hp = 100
    current_battle._team["charizard"]._current_hp = 50
    last_battle._team = {"charizard": Pokemon(species="charizard")}
    last_battle._team["charizard"]._max_hp = 100
    last_battle._team["charizard"]._current_hp = 100
    assert (
        SimpleRLAgent._calc_reward(last_battle, current_battle) == -MON_HP_REWARD * 50
    )
    last_battle._team["charizard"]._current_hp = 50
    current_battle._team["charizard"]._current_hp = 0
    current_battle._team["charizard"]._faint()
    assert (
        SimpleRLAgent._calc_reward(last_battle, current_battle)
        == -MON_HP_REWARD * 50 - MON_FAINTED_REWARD
    )


def test_calc_reward_opponent():
    current_battle = Battle("battle_tag", "username", None)  # noqa
    last_battle = Battle("battle_tag", "username", None)  # noqa
    current_battle._opponent_team = {"charizard": Pokemon(species="charizard")}
    current_battle._opponent_team["charizard"]._max_hp = 100
    current_battle._opponent_team["charizard"]._current_hp = 50
    last_battle._opponent_team = {"charizard": Pokemon(species="charizard")}
    last_battle._opponent_team["charizard"]._max_hp = 100
    last_battle._opponent_team["charizard"]._current_hp = 100
    assert SimpleRLAgent._calc_reward(last_battle, current_battle) == MON_HP_REWARD * 50
    last_battle._opponent_team["charizard"]._current_hp = 50
    current_battle._opponent_team["charizard"]._current_hp = 0
    current_battle._opponent_team["charizard"]._faint()
    assert (
        SimpleRLAgent._calc_reward(last_battle, current_battle)
        == MON_HP_REWARD * 50 + MON_FAINTED_REWARD
    )
    last_battle._opponent_team = {}
    assert (
        SimpleRLAgent._calc_reward(last_battle, current_battle)
        == MON_HP_REWARD * 100 + MON_FAINTED_REWARD
    )
    current_battle._opponent_team = {"charizard": Pokemon(species="charizard")}
    current_battle._opponent_team["charizard"]._max_hp = 100
    current_battle._opponent_team["charizard"]._current_hp = 50
    assert SimpleRLAgent._calc_reward(last_battle, current_battle) == MON_HP_REWARD * 50


def test_headers():
    agent = SimpleRLAgent(start_listening=False, battle_format="gen8randombattle")
    state_headers = agent._state_headers()
    action_headers = agent._action_space_headers()
    assert isinstance(state_headers, List)
    assert isinstance(action_headers, List)
    for h in state_headers:
        assert isinstance(h, str)
    for a in action_headers:
        assert isinstance(a, str)
    agent.b_format = "noformat"
    with pytest.raises(InvalidArgument):
        agent._state_headers()
    with pytest.raises(InvalidArgument):
        agent._action_space_headers()
