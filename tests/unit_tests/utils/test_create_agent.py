import pickle
import pytest

from io import BytesIO
from typing import List
from unittest.mock import patch

from agents.basic_rl import SimpleRLAgent
from agents.dad import Dad
from agents.eight_year_old_me import EightYearOldMe
from agents.expert_rl import ExpertRLAgent
from agents.sarsa_stark import SarsaStark, ExpertSarsaStark
from agents.advanced_heuristics import AdvancedHeuristics
from agents.twenty_year_old_me import TwentyYearOldMe
from utils.create_agent import create_agent, UnsupportedAgentType

_TEST_MODEL = {"test": 42}


def simulate_pickled_model():
    return BytesIO(pickle.dumps(_TEST_MODEL))


def get_mock_args():
    data = {
        "battle_format": "gen8randombattle",
        "start_listening": False,
        "max_concurrent_battles": 45,
    }
    return data


def check_agent_configuration(agent):
    assert agent.format == "gen8randombattle"
    with pytest.raises(AttributeError):
        agent._listening_coroutine  # noqa


def check_training_configuration(agent, keep_training):
    assert not agent.training
    assert agent.train_while_playing == keep_training
    assert agent.model == _TEST_MODEL


def test_invalid_cli_name():
    cli_name = "failure"
    with pytest.raises(UnsupportedAgentType):
        create_agent(cli_name, **get_mock_args())


def test_dad_creation():
    cli_name = "dad"
    agent = create_agent(cli_name, **get_mock_args())
    assert isinstance(agent, List)
    assert len(agent) == 1
    assert isinstance(agent[0], Dad)
    assert agent[0]._max_concurrent_battles == 45
    check_agent_configuration(agent[0])


def test_8_year_old_me_creation():
    cli_name = "8-year-old-me"
    agent = create_agent(cli_name, **get_mock_args())
    assert isinstance(agent, List)
    assert len(agent) == 1
    assert isinstance(agent[0], EightYearOldMe)
    assert agent[0]._max_concurrent_battles == 45
    check_agent_configuration(agent[0])


def test_20_year_old_me_creation():
    cli_name = "20-year-old-me"
    agent = create_agent(cli_name, **get_mock_args())
    assert isinstance(agent, List)
    assert len(agent) == 1
    assert isinstance(agent[0], TwentyYearOldMe)
    assert agent[0]._max_concurrent_battles == 45
    check_agent_configuration(agent[0])


def test_advanced_heuristics_creation():
    cli_name = "advanced-heuristics"
    agent = create_agent(cli_name, **get_mock_args())
    assert isinstance(agent, List)
    assert len(agent) == 1
    assert isinstance(agent[0], AdvancedHeuristics)
    assert agent[0]._max_concurrent_battles == 45
    check_agent_configuration(agent[0])


def test_simple_rl_player_best_creation():
    cli_name = "simpleRL-best"
    with patch("builtins.open") as mock_file:
        mock_file.return_value = simulate_pickled_model()
        agent = create_agent(cli_name, **get_mock_args())
        assert isinstance(agent, List)
        assert len(agent) == 1
        assert isinstance(agent[0], SimpleRLAgent)
        check_agent_configuration(agent[0])
        for a in agent:
            assert a._max_concurrent_battles == 45
            check_training_configuration(a, False)  # noqa


def test_simple_rl_player_best_train_creation():
    cli_name = "simpleRL-best-train"
    with patch("builtins.open") as mock_file:
        mock_file.return_value = simulate_pickled_model()
        agent = create_agent(cli_name, **get_mock_args())
        assert isinstance(agent, List)
        assert len(agent) == 1
        assert isinstance(agent[0], SimpleRLAgent)
        check_agent_configuration(agent[0])
        for a in agent:
            assert a._max_concurrent_battles == 1
            check_training_configuration(a, True)  # noqa


def test_simple_rl_player_all_creation():
    cli_name = "simpleRL-all"
    with patch("builtins.open") as mock_file:
        mock_file.side_effect = [simulate_pickled_model() for _ in range(2)]
        with patch("os.listdir") as mock_listdir:
            mock_listdir.return_value = ["test1", "test2"]
            agent = create_agent(cli_name, **get_mock_args())
            assert isinstance(agent, List)
            assert len(agent) == 2
            for a in agent:
                assert isinstance(a, SimpleRLAgent)
                assert a._max_concurrent_battles == 45
                check_agent_configuration(a)
                check_training_configuration(a, False)  # noqa


def test_simple_rl_player_all_train_creation():
    cli_name = "simpleRL-all-train"
    with patch("builtins.open") as mock_file:
        mock_file.side_effect = [simulate_pickled_model() for _ in range(2)]
        with patch("os.listdir") as mock_listdir:
            mock_listdir.return_value = ["test1", "test2"]
            agent = create_agent(cli_name, **get_mock_args())
            assert isinstance(agent, List)
            assert len(agent) == 2
            for a in agent:
                assert isinstance(a, SimpleRLAgent)
                assert a._max_concurrent_battles == 1
                check_agent_configuration(a)
                check_training_configuration(a, True)  # noqa


def test_expert_rl_player_best_creation():
    cli_name = "expertRL-best"
    with patch("builtins.open") as mock_file:
        mock_file.return_value = simulate_pickled_model()
        agent = create_agent(cli_name, **get_mock_args())
        assert isinstance(agent, List)
        assert len(agent) == 1
        assert isinstance(agent[0], ExpertRLAgent)
        check_agent_configuration(agent[0])
        for a in agent:
            assert a._max_concurrent_battles == 45
            check_training_configuration(a, False)  # noqa


def test_expert_rl_player_best_train_creation():
    cli_name = "expertRL-best-train"
    with patch("builtins.open") as mock_file:
        mock_file.return_value = simulate_pickled_model()
        agent = create_agent(cli_name, **get_mock_args())
        assert isinstance(agent, List)
        assert len(agent) == 1
        assert isinstance(agent[0], ExpertRLAgent)
        check_agent_configuration(agent[0])
        for a in agent:
            assert a._max_concurrent_battles == 1
            check_training_configuration(a, True)  # noqa


def test_expert_rl_player_all_creation():
    cli_name = "expertRL-all"
    with patch("builtins.open") as mock_file:
        mock_file.side_effect = [simulate_pickled_model() for _ in range(2)]
        with patch("os.listdir") as mock_listdir:
            mock_listdir.return_value = ["test1", "test2"]
            agent = create_agent(cli_name, **get_mock_args())
            assert isinstance(agent, List)
            assert len(agent) == 2
            for a in agent:
                assert isinstance(a, ExpertRLAgent)
                assert a._max_concurrent_battles == 45
                check_agent_configuration(a)
                check_training_configuration(a, False)  # noqa


def test_expert_rl_player_all_train_creation():
    cli_name = "expertRL-all-train"
    with patch("builtins.open") as mock_file:
        mock_file.side_effect = [simulate_pickled_model() for _ in range(2)]
        with patch("os.listdir") as mock_listdir:
            mock_listdir.return_value = ["test1", "test2"]
            agent = create_agent(cli_name, **get_mock_args())
            assert isinstance(agent, List)
            assert len(agent) == 2
            for a in agent:
                assert isinstance(a, ExpertRLAgent)
                assert a._max_concurrent_battles == 1
                check_agent_configuration(a)
                check_training_configuration(a, True)  # noqa


def test_simple_sarsa_stark_best_creation():
    cli_name = "simpleSarsaStark-best"
    with patch("builtins.open") as mock_file:
        mock_file.return_value = simulate_pickled_model()
        agent = create_agent(cli_name, **get_mock_args())
        assert isinstance(agent, List)
        assert len(agent) == 1
        assert isinstance(agent[0], SarsaStark)
        check_agent_configuration(agent[0])
        for a in agent:
            assert a._max_concurrent_battles == 45
            check_training_configuration(a, False)  # noqa


def test_simple_sarsa_stark_best_train_creation():
    cli_name = "simpleSarsaStark-best-train"
    with patch("builtins.open") as mock_file:
        mock_file.return_value = simulate_pickled_model()
        agent = create_agent(cli_name, **get_mock_args())
        assert isinstance(agent, List)
        assert len(agent) == 1
        assert isinstance(agent[0], SarsaStark)
        check_agent_configuration(agent[0])
        for a in agent:
            assert a._max_concurrent_battles == 1
            check_training_configuration(a, True)  # noqa


def test_simple_sarsa_stark_all_creation():
    cli_name = "simpleSarsaStark-all"
    with patch("builtins.open") as mock_file:
        mock_file.side_effect = [simulate_pickled_model() for _ in range(2)]
        with patch("os.listdir") as mock_listdir:
            mock_listdir.return_value = ["test1", "test2"]
            agent = create_agent(cli_name, **get_mock_args())
            assert isinstance(agent, List)
            assert len(agent) == 2
            for a in agent:
                assert isinstance(a, SarsaStark)
                assert a._max_concurrent_battles == 45
                check_agent_configuration(a)
                check_training_configuration(a, False)  # noqa


def test_simple_sarsa_stark_all_train_creation():
    cli_name = "simpleSarsaStark-all-train"
    with patch("builtins.open") as mock_file:
        mock_file.side_effect = [simulate_pickled_model() for _ in range(2)]
        with patch("os.listdir") as mock_listdir:
            mock_listdir.return_value = ["test1", "test2"]
            agent = create_agent(cli_name, **get_mock_args())
            assert isinstance(agent, List)
            assert len(agent) == 2
            for a in agent:
                assert isinstance(a, SarsaStark)
                assert a._max_concurrent_battles == 1
                check_agent_configuration(a)
                check_training_configuration(a, True)  # noqa


def test_expert_sarsa_stark_best_creation():
    cli_name = "expertSarsaStark-best"
    with patch("builtins.open") as mock_file:
        mock_file.return_value = simulate_pickled_model()
        agent = create_agent(cli_name, **get_mock_args())
        assert isinstance(agent, List)
        assert len(agent) == 1
        assert isinstance(agent[0], ExpertSarsaStark)
        check_agent_configuration(agent[0])
        for a in agent:
            assert a._max_concurrent_battles == 45
            check_training_configuration(a, False)  # noqa


def test_expert_sarsa_stark_best_train_creation():
    cli_name = "expertSarsaStark-best-train"
    with patch("builtins.open") as mock_file:
        mock_file.return_value = simulate_pickled_model()
        agent = create_agent(cli_name, **get_mock_args())
        assert isinstance(agent, List)
        assert len(agent) == 1
        assert isinstance(agent[0], ExpertSarsaStark)
        check_agent_configuration(agent[0])
        for a in agent:
            assert a._max_concurrent_battles == 1
            check_training_configuration(a, True)  # noqa


def test_expert_sarsa_stark_all_creation():
    cli_name = "expertSarsaStark-all"
    with patch("builtins.open") as mock_file:
        mock_file.side_effect = [simulate_pickled_model() for _ in range(2)]
        with patch("os.listdir") as mock_listdir:
            mock_listdir.return_value = ["test1", "test2"]
            agent = create_agent(cli_name, **get_mock_args())
            assert isinstance(agent, List)
            assert len(agent) == 2
            for a in agent:
                assert isinstance(a, ExpertSarsaStark)
                assert a._max_concurrent_battles == 45
                check_agent_configuration(a)
                check_training_configuration(a, False)  # noqa


def test_expert_sarsa_stark_all_train_creation():
    cli_name = "expertSarsaStark-all-train"
    with patch("builtins.open") as mock_file:
        mock_file.side_effect = [simulate_pickled_model() for _ in range(2)]
        with patch("os.listdir") as mock_listdir:
            mock_listdir.return_value = ["test1", "test2"]
            agent = create_agent(cli_name, **get_mock_args())
            assert isinstance(agent, List)
            assert len(agent) == 2
            for a in agent:
                assert isinstance(a, ExpertSarsaStark)
                assert a._max_concurrent_battles == 1
                check_agent_configuration(a)
                check_training_configuration(a, True)  # noqa
