import os
import pytest
import tensorflow as tf

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
from tf_agents.trajectories import TimeStep
from typing import Iterator, Union, List
from unittest.mock import create_autospec, patch, MagicMock, call

from agents.base_classes.tf_player import TFPlayer, _Env, _SavedPolicy


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


def test_saved_policy_init():
    with patch(
        "tf_agents.policies.py_tf_eager_policy.SavedModelPyTFEagerPolicy"
    ) as mock_saved_policy, patch("tensorflow.saved_model.load") as mock_load:
        test_path = "test path"
        mock_spec = MagicMock()
        mock_policy = MagicMock()
        mock_saved_policy.return_value = mock_spec
        mock_load.return_value = mock_policy

        policy = _SavedPolicy(model_path=test_path)

        assert policy.policy is mock_policy
        assert policy.time_step_spec is mock_spec.time_step_spec
        mock_saved_policy.assert_called_once_with(test_path, load_specs_from_pbtxt=True)
        mock_load.assert_called_once_with(test_path)


def test_saved_policy_action():
    with patch(
        "tf_agents.policies.py_tf_eager_policy.SavedModelPyTFEagerPolicy"
    ) as mock_saved_policy, patch("tensorflow.saved_model.load") as mock_load, patch(
        "agents.base_classes.tf_player._SavedPolicy.to_tensor"
    ) as mock_to_tensor:
        test_path = "test path"
        mock_spec = MagicMock()
        mock_policy = MagicMock()
        mock_saved_policy.return_value = mock_spec
        mock_load.return_value = mock_policy
        policy = _SavedPolicy(model_path=test_path)
        mock_to_tensor.return_value = "test"
        time_step = TimeStep("type", "reward", "discount", "observation")

        policy.action(time_step)

        mock_policy.action.assert_called_once_with(
            TimeStep("test", "test", "test", "test"), ()
        )
        assert mock_to_tensor.call_count == 4
        time_step_spec = mock_spec.time_step_spec
        calls = [
            call("type", time_step_spec.step_type),
            call("reward", time_step_spec.reward),
            call("discount", time_step_spec.discount),
            call("observation", time_step_spec.observation),
        ]
        mock_to_tensor.assert_has_calls(calls, any_order=True)


def test_saved_policy_to_tensor():
    start = {
        "obs1": tf.constant([0.1, 0.1, 0.2], dtype=tf.float64),
        "obs2": {
            "nested_obs1": tf.constant(1, dtype=tf.int32),
            "nested_obs2": {"further_nest": tf.constant([1, 2], dtype=tf.int64)},
        },
    }
    spec = {
        "obs1": tf.TensorSpec(shape=(3,), dtype=tf.float32),
        "obs2": {
            "nested_obs1": tf.TensorSpec(shape=(), dtype=tf.int64),
            "nested_obs2": {"further_nest": tf.TensorSpec(shape=(2,), dtype=tf.int64)},
        },
    }
    end = {
        "obs1": tf.constant([0.1, 0.1, 0.2], dtype=tf.float32),
        "obs2": {
            "nested_obs1": tf.constant(1, dtype=tf.int64),
            "nested_obs2": {"further_nest": tf.constant([1, 2], dtype=tf.int64)},
        },
    }

    to_tensor = _SavedPolicy.to_tensor(start, spec)

    assert tf.reduce_all(tf.equal(end["obs1"], to_tensor["obs1"]))
    assert tf.reduce_all(
        tf.equal(end["obs2"]["nested_obs1"], to_tensor["obs2"]["nested_obs1"])
    )
    assert tf.reduce_all(
        tf.equal(
            end["obs2"]["nested_obs2"]["further_nest"],
            to_tensor["obs2"]["nested_obs2"]["further_nest"],
        )
    )


def test_saved_policy_getattr():
    with patch(
        "tf_agents.policies.py_tf_eager_policy.SavedModelPyTFEagerPolicy"
    ) as mock_saved_policy, patch("tensorflow.saved_model.load") as mock_load:
        test_path = "test path"
        mock_spec = MagicMock()
        mock_policy = MagicMock()
        mock_saved_policy.return_value = mock_spec
        mock_load.return_value = mock_policy
        policy = _SavedPolicy(model_path=test_path)

        attribute = policy.attribute_that_does_not_exist
        policy.method_that_does_not_exist()

        assert attribute is mock_policy.attribute_that_does_not_exist
        mock_policy.method_that_does_not_exist.assert_called_once()


class AgentMock:
    policy = create_autospec(TFPolicy)

    def __init__(self):
        self.calls = 0

    def initialize(self):
        self.calls += 1

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

    def get_random_driver(self) -> PyDriver:
        assert self.agent == "Agent"
        assert self.replay_buffer == "Replay buffer"
        assert self.replay_buffer_iterator == "Iterator"
        return "Random driver"  # noqa: used for testing

    def get_collect_driver(self) -> PyDriver:
        assert self.agent == "Agent"
        assert self.replay_buffer == "Replay buffer"
        assert self.replay_buffer_iterator == "Iterator"
        assert self.random_driver == "Random driver"
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
        assert player.agent.calls == 1
        assert player.__getattr__("internal_agent") is None
        player.internal_agent = None
        assert player.battles is None
        assert player.format_is_doubles is None


def test_init_player_model_success():
    with patch(
        "tensorflow.saved_model.contains_saved_model"
    ) as mock_saved_model, patch("tensorflow.saved_model.load") as mock_load, patch(
        "os.path.isdir"
    ) as mock_isdir, patch(
        "tf_agents.environments.suite_gym.wrap_env"
    ) as mock_wrap, patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ) as mock_tf_wrap, patch(
        "tf_agents.policies.py_tf_eager_policy.SavedModelPyTFEagerPolicy"
    ) as mock_saved_policy, patch(
        "builtins.open"
    ) as mock_open, patch(
        "agents.base_classes.tf_player.load_code"
    ):
        mock_saved_model.return_value = True
        loaded_specs = MagicMock()
        mock_saved_policy.return_value = loaded_specs
        mock_load.return_value = AgentMock.policy
        mock_isdir.return_value = True
        player = DummyTFPlayer(
            "test path", start_listening=False, start_challenging=False, test=False
        )
        mock_saved_model.assert_called_once_with(os.path.join("test path", "model"))
        mock_isdir.assert_called_once_with(os.path.join("test path", "model"))
        mock_load.assert_called_once_with(os.path.join("test path", "model"))
        mock_open.assert_has_calls(
            [
                call(os.path.join("test path", "embed_battle_func.json")),
                call(os.path.join("test path", "embedding_description.json")),
            ],
            any_order=True,
        )
        assert player.policy.policy is AgentMock.policy
        assert player.policy.time_step_spec is loaded_specs.time_step_spec
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
    ) as mock_tf_wrap, patch(
        "builtins.open"
    ), patch(
        "agents.base_classes.tf_player.load_code"
    ):
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
        mock_isdir.assert_called_once_with(os.path.join("test path", "model"))
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
    ) as mock_tf_wrap, patch(
        "builtins.open"
    ), patch(
        "agents.base_classes.tf_player.load_code"
    ):
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
        mock_isdir.assert_called_once_with(os.path.join("test path", "model"))
        mock_saved_model.assert_called_once_with(os.path.join("test path", "model"))
        mock_load.assert_not_called()


def test_init_player_not_a_policy():
    with patch(
        "tensorflow.saved_model.contains_saved_model"
    ) as mock_saved_model, patch("os.path.isdir") as mock_isdir, patch(
        "tf_agents.environments.suite_gym.wrap_env"
    ) as mock_wrap, patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ) as mock_tf_wrap, patch(
        "agents.base_classes.tf_player._SavedPolicy"
    ) as mock_saved_policy, patch(
        "builtins.open"
    ), patch(
        "agents.base_classes.tf_player.load_code"
    ):
        mock_saved_model.return_value = True
        mock_saved_policy.return_value = "Not a policy"
        mock_isdir.return_value = True
        player = None
        with pytest.raises(RuntimeError):
            player = DummyTFPlayer(
                "test path", start_listening=False, start_challenging=False, test=False
            )
        assert player is None
        mock_wrap.assert_called_once()
        mock_tf_wrap.assert_called_once()
        mock_isdir.assert_called_once_with(os.path.join("test path", "model"))
        mock_saved_model.assert_called_once_with(os.path.join("test path", "model"))
        mock_saved_policy.assert_called_once_with(
            model_path=os.path.join("test path", "model")
        )


def test_save_policy_success():
    with patch("tf_agents.environments.suite_gym.wrap_env"), patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ), patch("tf_agents.policies.policy_saver.PolicySaver") as mock_saver, patch(
        "os.makedirs"
    ) as mock_makedirs, patch(
        "os.listdir"
    ) as mock_listdir, patch(
        "os.path.isdir"
    ) as mock_isdir, patch(
        "builtins.open"
    ) as mock_open:
        mock_saver_object = MagicMock()
        mock_saver.return_value = mock_saver_object
        mock_isdir.return_value = True
        mock_listdir.return_value = []
        player = DummyTFPlayer(
            start_listening=False, start_challenging=False, test=False
        )
        player.save_policy("save path")
        mock_isdir.assert_called_once_with("save path")
        mock_listdir.assert_called_once_with("save path")
        mock_makedirs.assert_has_calls(
            [call("save path", exist_ok=True), call(os.path.join("save path", "model"))]
        )
        mock_open.assert_has_calls(
            [
                call(os.path.join("save path", "embed_battle_func.json"), "w+"),
                call(os.path.join("save path", "embedding_description.json"), "w+"),
            ],
            any_order=True,
        )
        mock_saver_object.save.assert_called_once_with(
            os.path.join("save path", "model")
        )


def test_save_policy_failure():
    with patch("tf_agents.environments.suite_gym.wrap_env"), patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ), patch("tf_agents.policies.policy_saver.PolicySaver") as mock_saver, patch(
        "os.makedirs"
    ) as mock_makedirs, patch(
        "os.listdir"
    ) as mock_listdir, patch(
        "os.path.isdir"
    ) as mock_isdir:
        mock_saver_object = MagicMock()
        mock_saver.return_value = mock_saver_object
        mock_listdir.return_value = ["file"]
        mock_isdir.return_value = True
        player = DummyTFPlayer(
            start_listening=False, start_challenging=False, test=False
        )
        with pytest.raises(ValueError):
            player.save_policy("save path")
        mock_isdir.assert_called_once_with("save path")
        mock_listdir.assert_called_once_with("save path")
        mock_makedirs.assert_not_called()
        mock_saver_object.save.assert_not_called()


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
    ) as mock_checker, patch(
        "agents.base_classes.tf_player.RandomPlayer"
    ) as mock_random, patch(
        "agents.base_classes.tf_player.close_player"
    ) as mock_close:
        env1 = MagicMock()
        env2 = MagicMock()
        mock_opponent = MagicMock()
        mock_random.return_value = mock_opponent
        mock_env.side_effect = [env1, env2]
        player = DummyTFPlayer(
            start_listening=False, start_challenging=False, test=True
        )
        assert mock_env.call_count == 2
        mock_checker.assert_called_once_with(env1)
        mock_wrap.assert_called_once_with(env2)
        mock_close.assert_has_calls([call(env1.agent), call(mock_opponent)])
