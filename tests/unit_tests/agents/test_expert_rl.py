import pytest

from typing import List
from unittest.mock import patch

from agents.expert_rl import (
    ExpertRLAgent,
    _action_to_move_gen8random,
    _battle_to_state_gen8random,
)
from utils.invalid_argument import InvalidArgument


def test_get_battle_to_state_func_failure():
    with patch("agents.expert_rl.ExpertRLAgent._get_action_to_move_func"), patch(
        "agents.expert_rl.ExpertRLAgent._get_action_space_size"
    ):
        with pytest.raises(InvalidArgument):
            ExpertRLAgent(start_listening=False, battle_format="nobattleformat")


def test_get_battle_to_state_func_success():
    agent = ExpertRLAgent(start_listening=False, battle_format="gen8randombattle")
    assert agent._get_battle_to_state_func() is _battle_to_state_gen8random


def test_get_action_to_move_func_failure():
    with patch("agents.expert_rl.ExpertRLAgent._get_battle_to_state_func"), patch(
        "agents.expert_rl.ExpertRLAgent._get_action_space_size"
    ):
        with pytest.raises(InvalidArgument):
            ExpertRLAgent(start_listening=False, battle_format="nobattleformat")


def test_get_action_to_move_func_success():
    agent = ExpertRLAgent(start_listening=False, battle_format="gen8randombattle")
    assert agent._get_action_to_move_func() is _action_to_move_gen8random


def test_get_action_space_size_failure():
    with patch("agents.expert_rl.ExpertRLAgent._get_battle_to_state_func"), patch(
        "agents.expert_rl.ExpertRLAgent._get_action_to_move_func"
    ):
        with pytest.raises(InvalidArgument):
            ExpertRLAgent(start_listening=False, battle_format="nobattleformat")


def test_get_action_space_size_success():
    agent = ExpertRLAgent(start_listening=False, battle_format="gen8randombattle")
    assert agent._get_action_space_size() == 9


def test_headers():
    agent = ExpertRLAgent(start_listening=False, battle_format="gen8randombattle")
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
