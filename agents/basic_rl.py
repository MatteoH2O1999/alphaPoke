# AI trained with a simple RL algorithm
from typing import List
from poke_env.environment.abstract_battle import AbstractBattle

from . import VICTORY_REWARD, MON_HP_REWARD, MON_FAINTED_REWARD
from .trainable_player import TrainablePlayer
from utils import InvalidArgument
from utils.action_to_move_function import action_to_move_gen8single


class SimpleRLAgent(TrainablePlayer):
    def _get_battle_to_state_func(self):
        if self.b_format == "gen8randombattle":
            return _battle_to_state_gen8random
        else:
            raise InvalidArgument(
                f"{self.b_format} is not a valid battle format for this RL agent"
            )

    def _get_action_to_move_func(self):
        if self.b_format == "gen8randombattle":
            return action_to_move_gen8single
        else:
            raise InvalidArgument(
                f"{self.b_format} is not a valid battle format for this RL agent"
            )

    def _get_action_space_size(self):
        if self.b_format == "gen8randombattle":
            return 22
        else:
            raise InvalidArgument(
                f"{self.b_format} is not a valid battle format for this RL agent"
            )

    def _train(self, last_state, last_action, reward):
        model_to_edit = self.model[last_state]
        learning_rate = self._get_learning_rate(model_to_edit[2][last_action])
        model_to_edit[0][last_action] = model_to_edit[0][
            last_action
        ] + learning_rate * (reward - model_to_edit[0][last_action])
        model_to_edit[1] += 1
        model_to_edit[2][last_action] += 1

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
            if (
                not last_battle.team[mon_id].fainted
                and current_battle.team[mon_id].fainted
            ):
                reward -= MON_FAINTED_REWARD
                reward -= (
                    MON_HP_REWARD * last_battle.team[mon_id].current_hp_fraction * 100
                )
            elif (
                not last_battle.team[mon_id].fainted
                and not current_battle.team[mon_id].fainted
            ):
                diff = (
                    last_battle.team[mon_id].current_hp_fraction
                    - current_battle.team[mon_id].current_hp_fraction
                )
                reward -= MON_HP_REWARD * diff * 100
        for mon_id in opponent_mons_ids:
            if mon_id not in last_battle.opponent_team.keys():
                if current_battle.opponent_team[mon_id].fainted:
                    reward += MON_HP_REWARD * 100 + MON_FAINTED_REWARD
                else:
                    reward += MON_HP_REWARD * 100 * (
                        1 - current_battle.opponent_team[mon_id].current_hp_fraction
                    )
            else:
                if (
                    current_battle.opponent_team[mon_id].fainted
                    and not last_battle.opponent_team[mon_id].fainted
                ):
                    reward += MON_FAINTED_REWARD + (
                        100
                        * MON_HP_REWARD
                        * last_battle.opponent_team[mon_id].current_hp_fraction
                    )
                elif (
                    not current_battle.opponent_team[mon_id].fainted
                    and not last_battle.opponent_team[mon_id].fainted
                ):
                    diff = (
                        last_battle.opponent_team[mon_id].current_hp_fraction
                        - current_battle.opponent_team[mon_id].current_hp_fraction
                    )
                    reward += MON_HP_REWARD * 100 * diff
        return reward

    def _state_headers(self) -> List[str]:
        if self.b_format == "gen8randombattle":
            return [
                "Stat balance",
                "Type balance",
                "Boosts balance",
                "Is dynamaxed",
                "Forced switch",
                "Can apply status",
                "Can power up" "Move 1 value",
                "Move 2 value",
                "Move 3 value",
                "Move 4 value",
            ]
        else:
            raise InvalidArgument(
                f"{self.b_format} is not a valid battle format for this RL agent"
            )

    def _action_space_headers(self) -> List[str]:
        if self.b_format == "gen8randombattle":
            return [
                "Use move 1",
                "Use move 2",
                "Use move 3",
                "Use move 4",
                "Use move 1 and mega evolve",
                "Use move 2 and mega evolve",
                "Use move 3 and mega evolve",
                "Use move 4 and mega evolve",
                "Use move 1 as Z move",
                "Use move 2 as Z move",
                "Use move 3 as Z move",
                "Use move 4 as Z move",
                "Use move 1 and Dynamax",
                "Use move 2 and Dynamax",
                "Use move 3 and Dynamax",
                "Use move 4 and Dynamax",
                "Switch 1",
                "Switch 2",
                "Switch 3",
                "Switch 4",
                "Switch 5",
            ]
        else:
            raise InvalidArgument(
                f"{self.b_format} is not a valid battle format for this RL agent"
            )


def _battle_to_state_gen8random(battle: AbstractBattle):
    to_embed = []

    # Battle balance stats
    player_mon_stats = sum(battle.active_pokemon.base_stats.values())
    opponent_mon_stats = sum(battle.opponent_active_pokemon.base_stats.values())
    diff = player_mon_stats - opponent_mon_stats
    balance = 0
    if diff < -150:
        balance = -2
    elif diff < 0:
        balance = -1
    elif diff > 150:
        balance = 2
    elif diff > 0:
        balance = 1
    to_embed.append(balance)

    # Battle balance damage multipliers
    player_mon_types = battle.active_pokemon.types
    opponent_mon_types = battle.opponent_active_pokemon.types
    player_multiplier = 1.0
    opponent_multiplier = 1.0
    for t in player_mon_types:
        player_multiplier *= battle.opponent_active_pokemon.damage_multiplier(t)
    for t in opponent_mon_types:
        opponent_multiplier *= battle.active_pokemon.damage_multiplier(t)
    player_multiplier = round(player_multiplier)
    opponent_multiplier = round(opponent_multiplier)
    type_balance = player_multiplier - opponent_multiplier
    if type_balance > 0:
        type_balance = 1
    elif type_balance < 0:
        type_balance = -1
    else:
        type_balance = 0
    to_embed.append(type_balance)

    # Boosts balance
    boosts = 0
    boosts += sum(battle.active_pokemon.boosts.values())
    boosts -= sum(battle.opponent_active_pokemon.boosts.values())
    if boosts > 0:
        boosts = 1
    elif boosts < 0:
        boosts = -1
    to_embed.append(boosts)

    # Is dynamaxed
    is_dyna = 0
    if battle.active_pokemon.is_dynamaxed:
        is_dyna = 1
    to_embed.append(is_dyna)

    # Force switch
    forced_switch = 0
    if battle.force_switch:
        forced_switch = 1
    to_embed.append(forced_switch)

    # Move value
    for move in battle.active_pokemon.moves.values():
        move_value = 0
        if battle.opponent_active_pokemon.damage_multiplier(move.type) > 1.0:
            move_value += 1
        else:
            move_value -= 1
        if move.base_power > 80:
            move_value += 1
        if move.base_power == 0:
            move_value = 0
        to_embed.append(move_value)

    return tuple(to_embed)
