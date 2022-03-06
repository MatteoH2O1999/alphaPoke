from poke_env.environment.battle import Battle
from unittest.mock import patch, MagicMock

from agents.sarsa_stark import SarsaStark, ExpertSarsaStark


def test_choose_move():
    battle = Battle("battle_tag", "username", None)  # noqa
    with patch("agents.basic_rl.SimpleRLAgent.choose_move"):
        agent = SarsaStark(start_listening=False, battle_format="gen8randombattle")
        agent.choose_move(battle)
        assert agent.current_state is None
    with patch("agents.basic_rl.SimpleRLAgent.choose_move"):
        agent = SarsaStark(
            start_listening=False, training=True, battle_format="gen8randombattle"
        )
        agent._copy_battle = MagicMock()
        agent._copy_battle.return_value = battle
        agent.choose_move(battle)
        agent._copy_battle.assert_called_once_with(battle)
        assert agent.current_state is battle
    with patch("agents.basic_rl.SimpleRLAgent.choose_move"):
        agent = ExpertSarsaStark(
            start_listening=False, battle_format="gen8randombattle"
        )
        agent.choose_move(battle)
        assert agent.current_state is None
    with patch("agents.basic_rl.SimpleRLAgent.choose_move"):
        agent = ExpertSarsaStark(
            start_listening=False, training=True, battle_format="gen8randombattle"
        )
        agent._copy_battle = MagicMock()
        agent._copy_battle.return_value = battle
        agent.choose_move(battle)
        agent._copy_battle.assert_called_once_with(battle)
        assert agent.current_state is battle


def test_train_simple_sarsa_stark():
    agent = SarsaStark(start_listening=False, battle_format="gen8randombattle")
    current_battle = Battle("next_battle_tag", "username", None)  # noqa
    agent.current_state = current_battle
    agent._get_learning_rate = MagicMock()
    agent._get_learning_rate.return_value = 0.5
    state = (1, 0)
    agent._battle_to_state = MagicMock()
    agent._battle_to_state.return_value = state
    agent._choose_action = MagicMock()
    agent._choose_action.return_value = 1
    model = {(1, 0): [[1, 0], 3, [2, 1]], (0, 1): [[1, 0], 5, [3, 2]]}
    agent.model = model
    agent._train((0, 1), 0, 0)
    assert agent.model == {(1, 0): [[1, 0], 3, [2, 1]], (0, 1): [[0.5, 0], 6, [4, 2]]}
    agent._get_learning_rate.assert_called_once_with(3)
    agent._battle_to_state.assert_called_once_with(current_battle)
    agent._choose_action.assert_called_once_with(state)


def test_train_expert_sarsa_stark():
    agent = ExpertSarsaStark(start_listening=False, battle_format="gen8randombattle")
    current_battle = Battle("next_battle_tag", "username", None)  # noqa
    agent.current_state = current_battle
    agent._get_learning_rate = MagicMock()
    agent._get_learning_rate.return_value = 0.5
    state = (1, 0)
    agent._battle_to_state = MagicMock()
    agent._battle_to_state.return_value = state
    agent._choose_action = MagicMock()
    agent._choose_action.return_value = 1
    model = {(1, 0): [[1, 0], 3, [2, 1]], (0, 1): [[1, 0], 5, [3, 2]]}
    agent.model = model
    agent._train((0, 1), 0, 0)
    assert agent.model == {(1, 0): [[1, 0], 3, [2, 1]], (0, 1): [[0.5, 0], 6, [4, 2]]}
    agent._get_learning_rate.assert_called_once_with(3)
    agent._battle_to_state.assert_called_once_with(current_battle)
    agent._choose_action.assert_called_once_with(state)


def test_battle_finished_callback():
    with patch(
        "agents.trainable_player.TrainablePlayer._battle_finished_callback"
    ) as mock_battle_finished:
        agent1 = SarsaStark(start_listening=False, battle_format="gen8randombattle")
        agent2 = ExpertSarsaStark(
            start_listening=False, battle_format="gen8randombattle"
        )
        battle = Battle("next_battle_tag", "username", None)  # noqa
        agent1._battle_finished_callback(battle)
        mock_battle_finished.assert_called_once_with(battle)
        assert agent1.current_state is None
        mock_battle_finished.reset_mock()
        agent2._battle_finished_callback(battle)
        mock_battle_finished.assert_called_once_with(battle)
        assert agent2.current_state is None
