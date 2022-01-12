# AI trained with a simple RL algorithm
from typing import List
from poke_env.environment.abstract_battle import AbstractBattle

from . import VICTORY_REWARD, MON_HP_REWARD, MON_FAINTED_REWARD
from .trainable_player import TrainablePlayer
from cross_eval import InvalidArgument
from utils.action_to_move_function import action_to_move_gen8random


class SimpleRLAgent(TrainablePlayer):

    def _get_battle_to_state_func(self):
        if self.b_format == 'gen8randombattle':
            return _battle_to_state_gen8random
        else:
            raise InvalidArgument(f'{self.b_format} is not a valid battle format for this RL agent')

    def _get_action_to_move_func(self):
        if self.b_format == 'gen8randombattle':
            return action_to_move_gen8random
        else:
            raise InvalidArgument(f'{self.b_format} is not a valid battle format for this RL agent')

    def _get_action_space_size(self):
        if self.b_format == 'gen8randombattle':
            return 22
        else:
            raise InvalidArgument(f'{self.b_format} is not a valid battle format for this RL agent')

    def _train(self, last_state, last_action, reward):
        learning_rate = self._get_learning_rate(self.model[last_state][1])
        self.model[last_state][0][last_action] = (self.model[last_state][0][last_action] + learning_rate
                                                  * (reward - self.model[last_state][0][last_action]))
        self.model[last_state][1] += 1

    @staticmethod
    def _calc_reward(last_battle: AbstractBattle, current_battle: AbstractBattle):
        reward = 0
        if current_battle.won:
            reward += VICTORY_REWARD
        elif current_battle.lost:
            reward -= VICTORY_REWARD
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

    @staticmethod
    def _state_headers() -> List[str]:
        # TODO
        pass

    @staticmethod
    def _action_space_headers() -> List[str]:
        return ['Use move 1', 'Use move 2', 'Use move 3', 'Use move 4',
                'Use move 1 and mega evolve', 'Use move 2 and mega evolve', 'Use move 3 and mega evolve', 'Use move 4 and mega evolve',
                'Use move 1 as Z move', 'Use move 2 as Z move', 'Use move 3 as Z move', 'Use move 4 as Z move',
                'Use move 1 and Dynamax', 'Use move 2 and Dynamax', 'Use move 3 and Dynamax', 'Use move 4 and Dynamax',
                'Switch 1', 'Switch 2', 'Switch 3', 'Switch 4', 'Switch 5']


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
