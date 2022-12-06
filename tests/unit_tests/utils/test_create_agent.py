#
# A pokémon showdown battle-bot project based on reinforcement learning techniques.
# Copyright (C) 2022 Matteo Dell'Acqua
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import os
import pickle
import pytest

from io import BytesIO
from typing import List
from unittest.mock import call, patch

from agents.alpha_poke import AlphaPokeSingleBattleModelLoader
from agents.basic_rl import SimpleRLAgent
from agents.dad import Dad
from agents.eight_year_old_me import EightYearOldMe
from agents.expert_rl import ExpertRLAgent
from agents.sarsa_stark import SarsaStark, ExpertSarsaStark
from agents.advanced_heuristics import AdvancedHeuristics
from agents.twenty_year_old_me import TwentyYearOldMe
from utils.create_agent import create_agent, UnsupportedAgentType, MODELS_PATH

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


def test_alpha_poke_single_battle_creation():
    cli_name = "alphaPokeSingle-test_path"
    with patch(
        "tf_agents.policies.py_tf_eager_policy.SavedModelPyTFEagerPolicy"
    ) as mock_saved_policy, patch("tensorflow.saved_model.load") as mock_load, patch(
        "os.path.isdir"
    ) as mock_is_dir, patch(
        "tensorflow.saved_model.contains_saved_model"
    ) as mock_contains, patch(
        "builtins.open"
    ) as mock_open, patch(
        "agents.base_classes.tf_player.load_code"
    ), patch(
        "tf_agents.environments.suite_gym.wrap_env"
    ), patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ):
        mock_is_dir.return_value = True
        mock_contains.return_value = True
        agent = create_agent(cli_name, **get_mock_args())
        assert isinstance(agent, List)
        assert len(agent) == 1
        assert isinstance(agent[0], AlphaPokeSingleBattleModelLoader)
        assert agent[0]._max_concurrent_battles == 1
        check_agent_configuration(agent[0])
        mock_saved_policy.assert_called_once_with(
            os.path.join(MODELS_PATH, "tf_models", "test_path", "model"),
            load_specs_from_pbtxt=True,
        )
        mock_load.assert_called_once_with(
            os.path.join(MODELS_PATH, "tf_models", "test_path", "model")
        )
        mock_is_dir.assert_called_once_with(
            os.path.join(MODELS_PATH, "tf_models", "test_path", "model")
        )
        mock_contains.assert_called_once_with(
            os.path.join(MODELS_PATH, "tf_models", "test_path", "model")
        )
        mock_open.assert_has_calls(
            [
                call(
                    os.path.join(
                        MODELS_PATH, "tf_models", "test_path", "embed_battle_func.json"
                    )
                ),
                call(
                    os.path.join(
                        MODELS_PATH,
                        "tf_models",
                        "test_path",
                        "embedding_description.json",
                    )
                ),
            ],
            any_order=True,
        )
