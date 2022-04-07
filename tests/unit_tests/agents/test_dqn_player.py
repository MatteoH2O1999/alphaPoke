from gym.spaces import Box, Space
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.openai_api import ObservationType
from poke_env.player.player import Player
from tf_agents.agents import TFAgent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.drivers.py_driver import PyDriver
from tf_agents.policies import TFPolicy
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from typing import Iterator, Union, List
from unittest.mock import MagicMock, call, create_autospec, patch

from agents.base_classes.dqn_player import DQNPlayer


class DummyDQNPlayer(DQNPlayer):

    mock_agent: TFAgent
    mock_buffer: ReplayBuffer
    mock_iterator: Iterator
    mock_driver: PyDriver

    def eval_function(self, step):
        pass

    def log_function(self, step, loss_info: LossInfo):
        pass

    def calc_reward(
        self, last_battle: AbstractBattle, current_battle: AbstractBattle
    ) -> float:
        return 42.0

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        return [1]

    @property
    def embedding(self) -> Space:
        return Box(low=0.0, high=2.0, shape=(1,), dtype=int)

    @property
    def opponents(self) -> Union[Player, str, List[Player], List[str]]:
        return "opponent"

    def get_agent(self) -> TFAgent:
        return self.mock_agent

    def get_replay_buffer(self) -> ReplayBuffer:
        return self.mock_buffer

    def get_replay_buffer_iterator(self) -> Iterator:
        return self.mock_iterator

    def get_collect_driver(self) -> PyDriver:
        return self.mock_driver

    @property
    def log_interval(self) -> int:
        return 1

    @property
    def eval_interval(self) -> int:
        return 10


def test_dqn_player_init():
    with patch("tf_agents.environments.suite_gym.wrap_env") as mock_wrap, patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ) as mock_tf_wrap, patch(
        "tf_agents.policies.policy_saver.PolicySaver"
    ) as mock_saver:
        DummyDQNPlayer.mock_agent = MagicMock()
        mock_policy = create_autospec(TFPolicy)
        DummyDQNPlayer.mock_agent.policy = mock_policy
        DummyDQNPlayer.mock_driver = MagicMock()
        DummyDQNPlayer.mock_iterator = MagicMock()
        DummyDQNPlayer.mock_buffer = MagicMock()

        player = DummyDQNPlayer(start_listening=False, start_challenging=False)

        assert isinstance(player, DummyDQNPlayer)
        mock_wrap.assert_called_once()
        mock_tf_wrap.assert_called_once()
        mock_saver.assert_called_once_with(mock_policy)


def test_dqn_player_train():
    with patch("tf_agents.environments.suite_gym.wrap_env"), patch(
        "tf_agents.environments.tf_py_environment.TFPyEnvironment"
    ), patch("tf_agents.policies.policy_saver.PolicySaver"), patch(
        "tf_agents.drivers.py_driver.PyDriver"
    ) as mock_driver, patch(
        "tf_agents.policies.random_tf_policy.RandomTFPolicy"
    ) as mock_random_policy, patch(
        "tf_agents.policies.py_tf_eager_policy.PyTFEagerPolicy"
    ) as mock_eager_policy:
        DummyDQNPlayer.mock_agent = MagicMock()
        DummyDQNPlayer.mock_agent.train_step_counter.numpy.side_effect = [
            i for i in range(21)
        ]
        mock_train_data = MagicMock()
        DummyDQNPlayer.mock_agent.train.return_value = mock_train_data
        mock_policy = create_autospec(TFPolicy)
        DummyDQNPlayer.mock_agent.policy = mock_policy
        DummyDQNPlayer.mock_driver = MagicMock()
        DummyDQNPlayer.mock_driver.run.return_value = (1, 2)
        DummyDQNPlayer.mock_iterator = MagicMock()
        DummyDQNPlayer.mock_iterator.__next__.return_value = (3, 4)
        DummyDQNPlayer.mock_buffer = MagicMock()
        mock_random_driver = MagicMock()
        mock_driver.return_value = mock_random_driver
        mock_eager_policy_instance = MagicMock()
        mock_eager_policy.return_value = mock_eager_policy_instance
        mock_random_policy_instance = MagicMock()
        mock_random_policy.return_value = mock_random_policy_instance

        player = DummyDQNPlayer(start_listening=False, start_challenging=False)
        player.eval_function = MagicMock()
        player.log_function = MagicMock()
        player.train(20)

        assert player.eval_function.call_count == 3
        player.eval_function.assert_has_calls([call(0), call(10), call(20)])
        assert player.log_function.call_count == 20
        player.log_function.assert_has_calls(
            [call(i, mock_train_data) for i in range(1, 21)]
        )
        mock_driver.assert_called_once_with(
            player.environment,
            mock_eager_policy_instance,
            [player.replay_buffer.add_batch],
            max_steps=100,
        )
        mock_eager_policy.assert_called_once_with(
            mock_random_policy_instance, use_tf_function=True, batch_time_steps=False
        )
        mock_random_policy.assert_called_once_with(
            player.environment.time_step_spec(), player.environment.action_spec()
        )
