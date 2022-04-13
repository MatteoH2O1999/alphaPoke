import pytest

from gym import Space
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.status import Status
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.player.openai_api import ObservationType
from poke_env.player.player import Player
from tf_agents.agents import TFAgent
from tf_agents.drivers.py_driver import PyDriver
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.policies import TFPolicy
from typing import Iterator, Union, List
from unittest.mock import create_autospec, patch, MagicMock

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


class AgentMock:
    policy = create_autospec(TFPolicy)

    def __eq__(self, other):
        return other == "Agent"


class DummyTFPlayer(TFPlayer):
    def calc_reward(
        self, last_battle: AbstractBattle, current_battle: AbstractBattle
    ) -> float:
        return 42.0

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        return "Embedded battle"

    @property
    def embedding(self) -> Space:
        return "Embedding description"  # noqa: used for testing

    @property
    def opponents(self) -> Union[Player, str, List[Player], List[str]]:
        return "Opponent"

    def get_agent(self) -> TFAgent:
        return AgentMock()  # noqa: used for testing

    def get_replay_buffer(self) -> ReplayBuffer:
        assert self.agent == "Agent"
        return "Replay buffer"  # noqa: used for testing

    def get_replay_buffer_iterator(self) -> Iterator:
        assert self.agent == "Agent"
        assert self.replay_buffer == "Replay buffer"
        return "Iterator"

    def get_collect_driver(self) -> PyDriver:
        assert self.agent == "Agent"
        assert self.replay_buffer == "Replay buffer"
        assert self.replay_buffer_iterator == "Iterator"
        return "Collect driver"  # noqa: used for testing

    def log_function(self, *args, **kwargs):
        pass

    def eval_function(self, *args, **kwargs):
        pass

    @property
    def log_interval(self) -> int:
        return 10

    @property
    def eval_interval(self) -> int:
        return 100

    def train(self, num_iterations: int):
        pass


def test_init_player_for_training():
    with patch("tf_agents.environments.suite_gym.wrap_env") as mock_wrap, patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ) as mock_tf_wrap, patch(
        "tf_agents.policies.policy_saver.PolicySaver"
    ) as mock_saver:
        player = DummyTFPlayer(
            start_listening=False, start_challenging=False, test=False
        )
        assert isinstance(player, DummyTFPlayer)
        mock_wrap.assert_called_once()
        mock_tf_wrap.assert_called_once()
        mock_saver.assert_called_once_with(AgentMock.policy)
        assert player.can_train


def test_init_player_model_success():
    with patch(
        "tensorflow.saved_model.contains_saved_model"
    ) as mock_saved_model, patch("tensorflow.saved_model.load") as mock_load, patch(
        "os.path.isdir"
    ) as mock_isdir, patch(
        "tf_agents.environments.suite_gym.wrap_env"
    ) as mock_wrap, patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ) as mock_tf_wrap:
        mock_saved_model.return_value = True
        mock_load.return_value = AgentMock.policy
        mock_isdir.return_value = True
        player = DummyTFPlayer(
            "test path", start_listening=False, start_challenging=False, test=False
        )
        mock_saved_model.assert_called_once_with("test path")
        mock_isdir.assert_called_once_with("test path")
        mock_load.assert_called_once_with("test path")
        assert player.policy is AgentMock.policy
        assert isinstance(player, DummyTFPlayer)
        mock_wrap.assert_called_once()
        mock_tf_wrap.assert_called_once()
        assert not player.can_train


def test_init_player_not_a_dir():
    with patch(
        "tensorflow.saved_model.contains_saved_model"
    ) as mock_saved_model, patch("tensorflow.saved_model.load") as mock_load, patch(
        "os.path.isdir"
    ) as mock_isdir, patch(
        "tf_agents.environments.suite_gym.wrap_env"
    ) as mock_wrap, patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ) as mock_tf_wrap:
        mock_saved_model.return_value = True
        mock_load.return_value = AgentMock.policy
        mock_isdir.return_value = False
        player = None
        with pytest.raises(ValueError):
            player = DummyTFPlayer(
                "test path", start_listening=False, start_challenging=False, test=False
            )
        assert player is None
        mock_wrap.assert_called_once()
        mock_tf_wrap.assert_called_once()
        mock_isdir.assert_called_once_with("test path")
        mock_saved_model.assert_not_called()
        mock_load.assert_not_called()


def test_init_player_not_a_model():
    with patch(
        "tensorflow.saved_model.contains_saved_model"
    ) as mock_saved_model, patch("tensorflow.saved_model.load") as mock_load, patch(
        "os.path.isdir"
    ) as mock_isdir, patch(
        "tf_agents.environments.suite_gym.wrap_env"
    ) as mock_wrap, patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ) as mock_tf_wrap:
        mock_saved_model.return_value = False
        mock_load.return_value = AgentMock.policy
        mock_isdir.return_value = True
        player = None
        with pytest.raises(ValueError):
            player = DummyTFPlayer(
                "test path", start_listening=False, start_challenging=False, test=False
            )
        assert player is None
        mock_wrap.assert_called_once()
        mock_tf_wrap.assert_called_once()
        mock_isdir.assert_called_once_with("test path")
        mock_saved_model.assert_called_once_with("test path")
        mock_load.assert_not_called()


def test_init_player_not_a_policy():
    with patch(
        "tensorflow.saved_model.contains_saved_model"
    ) as mock_saved_model, patch("tensorflow.saved_model.load") as mock_load, patch(
        "os.path.isdir"
    ) as mock_isdir, patch(
        "tf_agents.environments.suite_gym.wrap_env"
    ) as mock_wrap, patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ) as mock_tf_wrap:
        mock_saved_model.return_value = True
        mock_load.return_value = "Not a policy"
        mock_isdir.return_value = True
        player = None
        with pytest.raises(RuntimeError):
            player = DummyTFPlayer(
                "test path", start_listening=False, start_challenging=False, test=False
            )
        assert player is None
        mock_wrap.assert_called_once()
        mock_tf_wrap.assert_called_once()
        mock_isdir.assert_called_once_with("test path")
        mock_saved_model.assert_called_once_with("test path")
        mock_load.assert_called_once_with("test path")


def test_save_policy():
    with patch("tf_agents.environments.suite_gym.wrap_env"), patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ), patch("tf_agents.policies.policy_saver.PolicySaver") as mock_saver:
        mock_saver_object = MagicMock()
        mock_saver.return_value = mock_saver_object
        player = DummyTFPlayer(
            start_listening=False, start_challenging=False, test=False
        )
        player.save_policy("save path")
        mock_saver_object.save.assert_called_once_with("save path")


def test_battle_format_properties_gen8_random_battle():
    with patch("tf_agents.environments.suite_gym.wrap_env"), patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ), patch("tf_agents.policies.policy_saver.PolicySaver"), patch(
        "agents.base_classes.tf_player.get_int_action_to_move"
    ) as mock_a2m, patch(
        "agents.base_classes.tf_player.get_int_action_space_size"
    ) as mock_space_size:
        mock_space_size.return_value = 42
        player = DummyTFPlayer(
            start_listening=False, start_challenging=False, test=False
        )
        mock_a2m.assert_called_once_with("gen8randombattle", False)
        mock_space_size.assert_called_once_with("gen8randombattle", False)


def test_action_to_move_function_gen8_vgc_2022():
    with patch("tf_agents.environments.suite_gym.wrap_env"), patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ), patch("tf_agents.policies.policy_saver.PolicySaver"), patch(
        "agents.base_classes.tf_player.get_int_action_to_move"
    ) as mock_a2m, patch(
        "agents.base_classes.tf_player.get_int_action_space_size"
    ) as mock_space_size:
        mock_space_size.return_value = 42
        player = DummyTFPlayer(
            start_listening=False,
            start_challenging=False,
            battle_format="gen8vgc2022",
            test=False,
        )
        mock_a2m.assert_called_once_with("gen8vgc2022", True)
        mock_space_size.assert_called_once_with("gen8vgc2022", True)


def test_create_evaluation_env():
    with patch("tf_agents.environments.suite_gym.wrap_env"), patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ) as mock_tf_wrap, patch("tf_agents.policies.policy_saver.PolicySaver"), patch(
        "agents.base_classes.tf_player._Env"
    ) as mock_env:
        mock_tf_wrap.side_effect = ["base env", "created env"]
        player = DummyTFPlayer(
            start_listening=False, start_challenging=False, test=False
        )
        env, agent = player.create_evaluation_env()
        assert player.environment == "base env"
        assert env == "created env"
        assert mock_env.call_count == 2
        assert agent is mock_env().agent


def test_reward_computing_helper():
    with patch("tf_agents.environments.suite_gym.wrap_env"), patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ), patch("tf_agents.policies.policy_saver.PolicySaver"):
        player = DummyTFPlayer(
            start_listening=False, battle_format="gen8randombattle", test=False
        )
        battle_1 = Battle("bat1", player.username, player.logger)
        battle_2 = Battle("bat2", player.username, player.logger)
        battle_3 = Battle("bat3", player.username, player.logger)
        battle_4 = Battle("bat4", player.username, player.logger)

        assert (
            player.reward_computing_helper(
                battle_1,
                fainted_value=0,
                hp_value=0,
                number_of_pokemons=4,
                starting_value=0,
                status_value=0,
                victory_value=1,
            )
            == 0
        )

        battle_1._won = True
        assert (
            player.reward_computing_helper(
                battle_1,
                fainted_value=0,
                hp_value=0,
                number_of_pokemons=4,
                starting_value=0,
                status_value=0,
                victory_value=1,
            )
            == 1
        )

        assert (
            player.reward_computing_helper(
                battle_2,
                fainted_value=0,
                hp_value=0,
                number_of_pokemons=4,
                starting_value=0.5,
                status_value=0,
                victory_value=5,
            )
            == -0.5
        )

        battle_2._won = False
        assert (
            player.reward_computing_helper(
                battle_2,
                fainted_value=0,
                hp_value=0,
                number_of_pokemons=4,
                starting_value=0,
                status_value=0,
                victory_value=5,
            )
            == -5
        )

        battle_3._team = {i: Pokemon(species="slaking") for i in range(4)}
        battle_3._opponent_team = {i: Pokemon(species="slowbro") for i in range(3)}

        battle_3._team[0].status = Status["FRZ"]
        battle_3._team[1]._current_hp = 100
        battle_3._team[1]._max_hp = 200
        battle_3._opponent_team[0].status = Status["FNT"]
        battle_3._opponent_team[1].status = Status["FNT"]

        # Opponent: two fainted, one full hp opponent
        # You: one half hp mon, one frozen mon
        assert (
            player.reward_computing_helper(
                battle_3,
                fainted_value=2,
                hp_value=3,
                number_of_pokemons=4,
                starting_value=0,
                status_value=0.25,
                victory_value=100,
            )
            == 2.25
        )

        battle_3._won = True
        assert (
            player.reward_computing_helper(
                battle_3,
                fainted_value=2,
                hp_value=3,
                number_of_pokemons=4,
                starting_value=0,
                status_value=0.25,
                victory_value=100,
            )
            == 100
        )

        battle_4._team, battle_4._opponent_team = (
            battle_3._opponent_team,
            battle_3._team,
        )
        assert (
            player.reward_computing_helper(
                battle_4,
                fainted_value=2,
                hp_value=3,
                number_of_pokemons=4,
                starting_value=0,
                status_value=0.25,
                victory_value=100,
            )
            == -2.25
        )


def test_play_episode():
    with patch("tf_agents.environments.suite_gym.wrap_env"), patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ), patch("tf_agents.policies.policy_saver.PolicySaver"):
        player = DummyTFPlayer(
            start_listening=False, start_challenging=False, test=False
        )
        mock_reset = MagicMock()
        player.environment.reset = mock_reset
        time_step = MagicMock()
        time_step.is_last.return_value = False
        mock_reset.return_value = time_step

        policy = MagicMock()
        player.policy = policy
        action = MagicMock()
        policy.action = action
        action_step = MagicMock()
        action.return_value = action_step
        action_step.action = 42

        mock_step = MagicMock()
        player.environment.step = mock_step
        end_time_step = MagicMock()
        mock_step.return_value = end_time_step
        end_time_step.is_last.return_value = True

        player.play_episode()

        mock_reset.assert_called_once()
        action.assert_called_once_with(time_step)
        mock_step.assert_called_once_with(42)


def test_test_env_false():
    with patch("tf_agents.environments.suite_gym.wrap_env") as mock_wrap, patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ), patch("tf_agents.policies.policy_saver.PolicySaver"), patch(
        "agents.base_classes.tf_player._Env"
    ) as mock_env, patch(
        "agents.base_classes.tf_player.check_env"
    ) as mock_checker:
        env1 = MagicMock()
        env2 = MagicMock()
        mock_env.side_effect = [env1, env2]
        player = DummyTFPlayer(
            start_listening=False, start_challenging=False, test=False
        )
        mock_env.assert_called_once()
        mock_checker.assert_not_called()
        mock_wrap.assert_called_once_with(env1)


def test_test_env_true():
    with patch("tf_agents.environments.suite_gym.wrap_env") as mock_wrap, patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ), patch("tf_agents.policies.policy_saver.PolicySaver"), patch(
        "agents.base_classes.tf_player._Env"
    ) as mock_env, patch(
        "agents.base_classes.tf_player.check_env"
    ) as mock_checker:
        env1 = MagicMock()
        env2 = MagicMock()
        mock_env.side_effect = [env1, env2]
        player = DummyTFPlayer(
            start_listening=False, start_challenging=False, test=True
        )
        assert mock_env.call_count == 2
        mock_checker.assert_called_once_with(env1)
        mock_wrap.assert_called_once_with(env2)
