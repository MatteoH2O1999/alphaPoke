# AI trained with a simple RL algorithm
import copy
import math
import random

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from . import LEARNING_RATE_WHILE_PLAYING, MIN_LEARNING_RATE_WHILE_TRAINING
from . import EPSILON_WHILE_TRAINING_AND_PLAYING, MIN_EPSILON_WHILE_TRAINING
from . import VICTORY_REWARD, MON_HP_REWARD, MON_FAINTED_REWARD
from cross_eval import InvalidArgument
from utils import get_a2m_function, init_action_space


class SimpleRLAgent(Player):

    def __init__(self, **kwargs):
        self.model = kwargs.get('model', {})
        self.training = kwargs.get('training', False)
        self.train_while_playing = kwargs.get('keep_training', False)
        self.action_to_move_function = get_a2m_function(kwargs['battle_format'])
        self.b_format = kwargs.get('battle_format')
        self.battle_to_state_func = self.get_battle_to_state_func()
        self.last_state = None
        self.last_action = None
        if 'training' in kwargs.keys():
            kwargs.pop('training')
        if 'model' in kwargs.keys():
            kwargs.pop('model')
        if 'keep_training' in kwargs.keys():
            kwargs.pop('keep_training')
        if (self.training or self.train_while_playing) and 'max_concurrent_battles' in kwargs.keys():
            kwargs.pop('max_concurrent_battles')
        super().__init__(**kwargs)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if not isinstance(battle, Battle):
            raise RuntimeError('Error with battle transfer')
        if self.last_state and (self.training or self.train_while_playing):
            reward = self._calc_reward(self.last_state, battle)
            self._train(self._battle_to_state(self.last_state), self.last_action, reward)
        current_state = self._battle_to_state(battle)
        action = self._choose_action(current_state)
        if self.training or self.train_while_playing:
            self.last_state = self._copy_battle(battle)
            self.last_action = action
        return self._action_to_move(action, battle)

    def _choose_action(self, state):
        if state not in self.model.keys():
            self.model[state] = [init_action_space(self.b_format), 0]
        action_space = self.model[state][0]
        epsilon = self._get_epsilon(self.model[state][1])
        max_value = max(action_space)
        i = 0
        possible_indexes = []
        for action_value in action_space:
            if not action_value < max_value:
                possible_indexes.append(i)
            i += 1
        optimal_action = random.choice(possible_indexes)
        if random.random() < epsilon:
            tmp = optimal_action
            while optimal_action == tmp:
                optimal_action = random.randint(0, len(action_space) - 1)
        return optimal_action

    def _action_to_move(self, action: int, battle: Battle) -> BattleOrder:
        return self.action_to_move_function(self, action, battle)

    def get_battle_to_state_func(self):
        if self.b_format == 'gen8randombattle':
            return _battle_to_state_gen8random
        else:
            raise InvalidArgument(f'{self.b_format} is not a valid battle format for this RL agent')

    def _battle_to_state(self, battle: AbstractBattle):
        return self.battle_to_state_func(battle)

    def _train(self, last_state, last_action, reward):
        learning_rate = self._get_learning_rate(self.model[last_state][1])
        self.model[last_state][0][last_action] = (self.model[last_state][0][last_action] + learning_rate
                                                  * (reward - self.model[last_state][0][last_action]))
        self.model[last_state][1] += 1

    @staticmethod
    def _calc_reward(last_battle: AbstractBattle, current_battle: AbstractBattle):
        reward = 0
        player_mons_ids = list(current_battle.team.keys())
        opponent_mons_ids = list(current_battle.opponent_team.keys())
        for mon_id in player_mons_ids:
            if not last_battle.team[mon_id].fainted and current_battle.team[mon_id].fainted:
                reward -= MON_FAINTED_REWARD
                reward -= (MON_HP_REWARD * last_battle.team[mon_id].current_hp_fraction * 100)
            elif not last_battle.team[mon_id].fainted and not current_battle.team[mon_id].fainted:
                diff = (last_battle.team[mon_id].current_hp_fraction - current_battle.team[mon_id].current_hp_fraction)
                reward -= (MON_HP_REWARD * diff * 100)
        for mon_id in opponent_mons_ids:
            if mon_id not in last_battle.opponent_team.keys():
                if current_battle.opponent_team[mon_id].fainted:
                    reward += (MON_HP_REWARD * 100 + MON_FAINTED_REWARD)
                else:
                    reward += (MON_HP_REWARD * (100 - current_battle.opponent_team[mon_id].current_hp_fraction))
            else:
                if current_battle.opponent_team[mon_id].fainted and not last_battle.opponent_team[mon_id].fainted:
                    reward += (MON_FAINTED_REWARD + (100 * MON_HP_REWARD
                                                     * last_battle.opponent_team[mon_id].current_hp_fraction))
                elif not current_battle.opponent_team[mon_id].fainted and not last_battle.opponent_team[mon_id].fainted:
                    diff = (last_battle.opponent_team[mon_id].current_hp_fraction
                            - current_battle.opponent_team[mon_id].current_hp_fraction)
                    reward += (MON_HP_REWARD * 100 * diff)
        return reward

    def _get_epsilon(self, samples):
        epsilon = 0
        if self.training:
            epsilon = max(min(1 / math.log(max(2, samples)), 1.0), MIN_EPSILON_WHILE_TRAINING)
        elif self.train_while_playing:
            epsilon = EPSILON_WHILE_TRAINING_AND_PLAYING
        return epsilon

    def _get_learning_rate(self, samples):
        learning_rate = 0
        if self.training:
            learning_rate = max(1.0, min(80 / max(1, samples), MIN_LEARNING_RATE_WHILE_TRAINING))
        elif self.train_while_playing:
            learning_rate = LEARNING_RATE_WHILE_PLAYING
        return learning_rate

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        if self.training or self.train_while_playing:
            reward = self._calc_reward(self.last_state, battle)
            if not battle.finished:
                raise RuntimeError('???')
            if battle.won:
                reward += VICTORY_REWARD
            elif battle.lost:
                reward -= VICTORY_REWARD
            if self.last_state and (self.training or self.train_while_playing):
                self._train(self._battle_to_state(self.last_state), self.last_action, reward)
            self.last_state = None
            self.last_action = None

    @staticmethod
    def _copy_battle(battle):
        return copy.deepcopy(battle)

    def get_model(self):
        return self.model

    def get_pretty_model(self):
        return self._model_to_table(self.model)

    def reset_rates(self):
        for values in self.model.values():
            values[1] = 0

    @staticmethod
    def _model_to_table(model):
        # TODO
        return model


def _battle_to_state_gen8random(battle: AbstractBattle):
    to_embed = []
    player_mons_ids = list(battle.team.keys())
    opponent_mon = battle.opponent_active_pokemon
    opponent_hp = opponent_mon.current_hp_fraction
    opponent_status = 3
    if 1 > opponent_hp > 0.66:
        opponent_status = 2
    elif opponent_hp > 0.33:
        opponent_status = 1
    elif opponent_hp > 0:
        opponent_status = 0
    elif opponent_mon.fainted:
        opponent_status = -1
    to_embed.append(opponent_status)
    opponent_status_effect = 0
    if opponent_mon.status:
        opponent_status_effect = opponent_mon.status
    to_embed.append(opponent_status_effect)
    player_status_effect = 0
    if battle.active_pokemon.status:
        player_status_effect = battle.active_pokemon.status
    to_embed.append(player_status_effect)
    for mon_id in player_mons_ids:
        mon_status = 3
        mon_hp_fraction = battle.team[mon_id].current_hp_fraction
        if 1 > mon_hp_fraction > 0.66:
            mon_status = 2
        elif mon_hp_fraction > 0.33:
            mon_status = 1
        elif mon_hp_fraction > 0:
            mon_status = 0
        elif battle.team[mon_id].fainted:
            mon_status = -1
        to_embed.append(mon_status)
        multiplier = 1.0
        opponent_types = opponent_mon.types
        for mon_type in opponent_types:
            multiplier *= battle.team[mon_id].damage_multiplier(mon_type)
        to_embed.append(multiplier)
        stats_total = sum(battle.team[mon_id].base_stats.values())
        opponent_stats_total = sum(opponent_mon.base_stats.values())
        diff = stats_total - opponent_stats_total
        stats_comparison = 0
        if diff < -150:
            stats_comparison = -2
        elif diff < 0:
            stats_comparison = -1
        elif diff > 150:
            stats_comparison = 2
        elif diff > 0:
            stats_comparison = 1
        to_embed.append(stats_comparison)
    for move in battle.active_pokemon.moves.values():
        to_embed.append(opponent_mon.damage_multiplier(move))
        move_power = move.base_power
        move_category = 0
        if 0 < move_power < 40:
            move_category = 1
        elif move_power < 75:
            move_category = 2
        elif move_power < 100:
            move_category = 3
        elif move_power >= 100:
            move_category = 4
        to_embed.append(move_category)
    forced_switch = 0
    if battle.force_switch:
        forced_switch = 1
    to_embed.append(forced_switch)
    is_dyna = 0
    if battle.active_pokemon.is_dynamaxed:
        is_dyna = 1
    to_embed.append(is_dyna)
    can_dyna = 0
    if battle.can_dynamax:
        can_dyna = 1
    to_embed.append(can_dyna)
    can_mega = 0
    if battle.can_mega_evolve:
        can_mega = 1
    to_embed.append(can_mega)
    can_z = 0
    if battle.can_z_move:
        can_z = 1
    to_embed.append(can_z)
    return tuple(to_embed)
