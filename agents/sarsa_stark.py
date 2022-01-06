# AI trained with SARSA algorithm
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from . import LEARNING_RATE_WHILE_PLAYING


class SarsaStark(Player):

    def __init__(self, **kwargs):
        self.model = kwargs.get('model', {})
        self.training = kwargs.get('training', False)
        self.train_while_playing = kwargs.get('keep_training', False)
        self.static_epsilon = LEARNING_RATE_WHILE_PLAYING
        if self.training:
            self.training_steps = 0
            self.last_state = None
            self.last_action = None
        super().__init__(**kwargs)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        pass
