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
from unittest.mock import MagicMock, create_autospec, patch

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
