import pytest

from gym import Space
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
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
        player = DummyTFPlayer(start_listening=False, start_challenging=False)
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
            "test path", start_listening=False, start_challenging=False
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
                "test path", start_listening=False, start_challenging=False
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
                "test path", start_listening=False, start_challenging=False
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
                "test path", start_listening=False, start_challenging=False
            )
        assert player is None
        mock_wrap.assert_called_once()
        mock_tf_wrap.assert_called_once()
        mock_isdir.assert_called_once_with("test path")
        mock_saved_model.assert_called_once_with("test path")
        mock_load.assert_called_once_with("test path")


def test_save_policy():
    with patch("tf_agents.environments.suite_gym.wrap_env") as mock_wrap, patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ) as mock_tf_wrap, patch(
        "tf_agents.policies.policy_saver.PolicySaver"
    ) as mock_saver:
        mock_saver_object = MagicMock()
        mock_saver.return_value = mock_saver_object
        player = DummyTFPlayer(start_listening=False, start_challenging=False)
        player.save_policy("save path")
        mock_saver_object.save.assert_called_once_with("save path")


def test_battle_format_properties_gen8_random_battle():
    with patch("tf_agents.environments.suite_gym.wrap_env") as mock_wrap, patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ) as mock_tf_wrap, patch(
        "tf_agents.policies.policy_saver.PolicySaver"
    ) as mock_saver, patch(
        "agents.base_classes.tf_player.get_int_action_to_move"
    ) as mock_a2m, patch(
        "agents.base_classes.tf_player.get_int_action_space_size"
    ) as mock_space_size:
        mock_space_size.return_value = 42
        player = DummyTFPlayer(start_listening=False, start_challenging=False)
        mock_a2m.assert_called_once_with("gen8randombattle", False)
        mock_space_size.assert_called_once_with("gen8randombattle", False)


def test_action_to_move_function_gen8_vgc_2022():
    with patch("tf_agents.environments.suite_gym.wrap_env") as mock_wrap, patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ) as mock_tf_wrap, patch(
        "tf_agents.policies.policy_saver.PolicySaver"
    ) as mock_saver, patch(
        "agents.base_classes.tf_player.get_int_action_to_move"
    ) as mock_a2m, patch(
        "agents.base_classes.tf_player.get_int_action_space_size"
    ) as mock_space_size:
        mock_space_size.return_value = 42
        player = DummyTFPlayer(
            start_listening=False, start_challenging=False, battle_format="gen8vgc2022"
        )
        mock_a2m.assert_called_once_with("gen8vgc2022", True)
        mock_space_size.assert_called_once_with("gen8vgc2022", True)
