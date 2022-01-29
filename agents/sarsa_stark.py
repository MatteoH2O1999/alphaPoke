# AI trained with SARSA or Q-learning algorithm
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder

from .basic_rl import SimpleRLAgent
from .expert_rl import ExpertRLAgent


class SarsaStark(SimpleRLAgent):

    def __init__(self, **kwargs):
        self.current_state = None
        super().__init__(**kwargs)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        self.current_state = self._copy_battle(battle)
        return super().choose_move(battle)

    def _train(self, last_state, last_action, reward):
        model_to_edit = self.model[last_state]
        learning_rate = self._get_learning_rate(model_to_edit[2][last_action])
        current_state = self._battle_to_state(self.current_state)
        next_action = self._choose_action(current_state)
        model_to_edit[0][last_action] = (model_to_edit[0][last_action]
                                         + (learning_rate * (reward + self.model[current_state][0][next_action]
                                                             - model_to_edit[0][last_action])))
        model_to_edit[1] += 1
        model_to_edit[2][last_action] += 1

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        self.current_state = self._copy_battle(battle)
        super()._battle_finished_callback(battle)
        self.current_state = None


class ExpertSarsaStark(ExpertRLAgent):

    def __init__(self, **kwargs):
        self.current_state = None
        super().__init__(**kwargs)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        self.current_state = self._copy_battle(battle)
        return super().choose_move(battle)

    def _train(self, last_state, last_action, reward):
        model_to_edit = self.model[last_state]
        learning_rate = self._get_learning_rate(model_to_edit[2][last_action])
        current_state = self._battle_to_state(self.current_state)
        next_action = self._choose_action(current_state)
        model_to_edit[0][last_action] = (model_to_edit[0][last_action]
                                         + (learning_rate * (reward + self.model[current_state][0][next_action]
                                                             - model_to_edit[0][last_action])))
        model_to_edit[1] += 1
        model_to_edit[2][last_action] += 1

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        self.current_state = self._copy_battle(battle)
        super()._battle_finished_callback(battle)
        self.current_state = None
