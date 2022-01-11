# AI trained with simple RL algorithm but with a simplified action space:
# 1: fight to kill
# 2: fight with weak move
# 3: fight with move to apply status effect and damage
# 4: power up
# 5: apply status effect
# 6: sacrifice (switch to a pokémon with low health or really weak)
# 7: defensive switch (switch to a pokémon resistant to the enemy's active pokémon)
# 8: offensive switch (switch to a pokémon supereffective against the enemy's active pokémon or really strong)
# 9: fight predict (use a move predicting a switch)
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from cross_eval import InvalidArgument
from .basic_rl import SimpleRLAgent


class ExpertRLAgent(SimpleRLAgent):

    def _get_action_to_move_func(self):
        if self.b_format == 'gen8randombattle':
            return _action_to_move_gen8random
        else:
            raise InvalidArgument(f'{self.b_format} is not a valid battle format for this RL agent')

    def _get_action_space_size(self):
        if self.b_format == 'gen8randombattle':
            return 9
        else:
            raise InvalidArgument(f'{self.b_format} is not a valid battle format for this RL agent')

    @staticmethod
    def _model_to_table(model):
        # TODO
        return model


def _action_to_move_gen8random(agent: Player, action: int, battle: Battle) -> BattleOrder:
    if action == 0:
        return _fight_to_kill(agent, battle)
    elif action == 1:
        return _fight_weak_move(agent, battle)
    elif action == 2:
        return _fight_status_effect(agent, battle)
    elif action == 3:
        return _power_up(agent, battle)
    elif action == 4:
        return _status_effect(agent, battle)
    elif action == 5:
        return _sac(agent, battle)
    elif action == 6:
        return _defensive_switch(agent, battle)
    elif action == 7:
        return _offensive_switch(agent, battle)
    elif action == 8:
        return _fight_predict(agent, battle)
    else:
        raise RuntimeError('???')


# 1: fight to kill
def _fight_to_kill(agent: Player, battle: Battle):
    pass


# 2: fight with weak move
def _fight_weak_move(agent: Player, battle: Battle):
    pass


# 3: fight with move to apply status effect and damage
def _fight_status_effect(agent: Player, battle: Battle):
    pass


# 4: power up
def _power_up(agent: Player, battle: Battle):
    pass


# 5: apply status effect
def _status_effect(agent: Player, battle: Battle):
    pass


# 6: sacrifice (switch to a pokémon with low health or really weak)
def _sac(agent: Player, battle: Battle):
    pass


# 7: defensive switch (switch to a pokémon resistant to the enemy's active pokémon)
def _defensive_switch(agent: Player, battle: Battle):
    pass


# 8: offensive switch (switch to a pokémon supereffective against the enemy's active pokémon or really strong)
def _offensive_switch(agent: Player, battle: Battle):
    pass


# 9: fight predict (use a move predicting a switch)
def _fight_predict(agent: Player, battle: Battle):
    pass
