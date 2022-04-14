# Player with advanced heuristics from friend
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Gen8Battle
from poke_env.environment.move import Move
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.status import Status
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player
from poke_env.utils import compute_raw_stats
from typing import Awaitable, Optional, Union

from utils.get_smogon_data import get_random_battle_learnset

BOOSTS = [0.25, 0.28, 0.33, 0.4, 0.5, 0.66, 1, 1.5, 2, 2.5, 3, 3.5, 4]

BOOST_THRESHOLD = 3

THRESHOLDS = {
    "hp": 80,
    "atk": 30,
    "def": 80,
    "spa": 30,
    "spd": 80,
    "spe": 75,
}


class Seba(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.format == "gen8randombattle":
            self.learnset = get_random_battle_learnset(8)

    def choose_move(
        self, battle: AbstractBattle
    ) -> Union[BattleOrder, Awaitable[BattleOrder]]:
        if isinstance(battle, Gen8Battle):
            return self.choose_move_gen8_random_battle(battle)
        else:
            raise NotImplementedError(
                f"{self.format} is not supported yet as it requires support for type {type(battle)}."
            )

    def choose_move_gen8_random_battle(
        self, battle: Gen8Battle
    ) -> Union[BattleOrder, Awaitable[BattleOrder]]:
        if battle.force_switch:
            return self.switch(battle)
        if self.is_faster(battle.active_pokemon, battle.opponent_active_pokemon):
            if self.has_stab_super_effective(battle):
                if self.do_i_one_shot(battle):
                    if self.should_i_one_shot(battle):
                        return self.create_order(self.get_stab_super_effective(battle))
                    else:
                        return self.create_order(self.boost(battle))
                else:
                    if self.can_i_bi_shot(battle):
                        self.create_order(self.get_bi_shot_move(battle))
                    else:
                        if self.can_status(battle):
                            return self.create_order(self.get_status(battle))
                        else:
                            return self.switch(battle)
            else:
                if self.resist(battle):
                    return self.create_order(self.get_resisted_move(battle))
                else:
                    if self.can_status(battle):
                        return self.create_order(self.get_status(battle))
                    else:
                        return self.switch(battle)
        else:
            if self.do_i_get_oneshot(battle):
                return self.switch(battle)
            else:
                if self.do_i_one_shot(battle):
                    return self.create_order(self.get_stab_super_effective(battle))
                else:
                    # TODO
                    pass
        return self.choose_random_move(battle)

    def switch(self, battle: Gen8Battle) -> BattleOrder:
        return self.choose_random_move(battle)

    def is_faster(self, mon: Pokemon, enemy: Pokemon) -> bool:
        my_speed = mon.stats["spe"]
        speed_boost = 6
        speed_boost += mon.boosts["spe"]
        my_speed *= BOOSTS[speed_boost]
        enemy_base_speed = enemy.base_stats["spe"]
        if enemy_base_speed > THRESHOLDS["spe"]:
            enemy_speed = compute_raw_stats(
                enemy.species,
                [0, 0, 0, 0, 0, 252],
                [31, 31, 31, 31, 31, 31],
                enemy.level,
                "hardy",
            )[5]
        else:
            enemy_speed = compute_raw_stats(
                enemy.species,
                [0, 0, 0, 0, 0, 0],
                [31, 31, 31, 31, 31, 31],
                enemy.level,
                "hardy",
            )[5]
        if "Choice Scarf" in self.learnset[enemy.species]["items"]:
            enemy_speed *= 1.5
        enemy_boost = 6
        enemy_boost += enemy.boosts["spe"]
        enemy_speed *= BOOSTS[enemy_boost]
        return my_speed > enemy_speed

    @staticmethod
    def get_stab_super_effective(battle: Gen8Battle) -> Optional[Move]:
        mon = battle.active_pokemon
        enemy = battle.opponent_active_pokemon
        possible_moves = []
        for move in mon.moves.values():
            stab = move.type in mon.types
            damage_multiplier = enemy.damage_multiplier(move)
            if stab and damage_multiplier >= 2.0 and move in battle.available_moves:
                possible_moves.append(move)
        chosen_move = None
        for move in possible_moves:
            if chosen_move is None or move.base_power > chosen_move.base_power:
                chosen_move = move
        return chosen_move

    def has_stab_super_effective(self, battle: Gen8Battle) -> bool:
        return self.get_stab_super_effective(battle) is not None

    def do_i_one_shot(self, battle: Gen8Battle) -> bool:
        enemy = battle.opponent_active_pokemon
        if not self.has_stab_super_effective(battle):
            return False
        move = self.get_stab_super_effective(battle)
        if move.base_power < 80:
            return False
        move_type = move.category
        if (
            move_type == MoveCategory.PHYSICAL
            and enemy.base_stats["def"] + enemy.base_stats["hp"]
            > THRESHOLDS["hp"] + THRESHOLDS["def"]
        ):
            return False
        if (
            move_type == MoveCategory.SPECIAL
            and enemy.base_stats["spd"] + enemy.base_stats["hp"]
            > THRESHOLDS["hp"] + THRESHOLDS["spd"]
        ):
            return False
        if (
            "Focus Sash" in self.learnset[enemy.species]["items"]
            or "Sturdy" in self.learnset[enemy.species]["abilities"]
        ):
            return False
        return True

    def should_i_boost(self, battle: Gen8Battle) -> bool:
        mon = battle.active_pokemon
        enemy = battle.opponent_active_pokemon
        can_boost = False
        for move in mon.moves.values():
            if move.self_boost is not None:
                if sum(move.self_boost.values()) > 0 and move in battle.available_moves:
                    can_boost = True
        if not can_boost:
            return False
        tmp = self.learnset[enemy.species]["moves"]
        opponent_possible_moves = []
        for m in tmp:
            opponent_possible_moves.append(
                Move(m.lower().replace(" ", "").replace("-", "").replace("'", ""))
            )
        for move in opponent_possible_moves:
            if (
                mon.damage_multiplier(move) >= 2.0
                and move.base_power >= 40
                and move.expected_hits <= 1.0
            ):
                return False
            if (
                1.0 <= mon.damage_multiplier(move) < 2.0
                and move.type in enemy.types
                and move.base_power >= 80
            ):
                return False
        current_boosts = sum(mon.boosts.values())
        if current_boosts > BOOST_THRESHOLD:
            return False
        return True

    def should_i_one_shot(self, battle: Gen8Battle) -> bool:
        return not self.should_i_boost(battle)

    @staticmethod
    def boost(battle: Gen8Battle) -> Move:
        mon = battle.active_pokemon
        for move in mon.moves.values():
            if sum(move.self_boost.values()) > 0 and move in battle.available_moves:
                return move

    def can_i_bi_shot(self, battle: Gen8Battle) -> bool:
        return self.get_bi_shot_move(battle) is not None

    def get_bi_shot_move(self, battle: Gen8Battle) -> Optional[Move]:
        if self.do_i_get_oneshot(battle):
            return None
        mon = battle.active_pokemon
        enemy = battle.opponent_active_pokemon
        count = 4
        for move in mon.moves.values():
            if enemy.damage_multiplier(move) < 2.0 and move.base_power < 85:
                count -= 1
        if count == 0:
            return None
        possible_moves = []
        for move in mon.moves.values():
            move_type = move.category
            if move_type == MoveCategory.PHYSICAL and enemy.base_stats[
                "hp"
            ] + enemy.base_stats["def"] > 2.5 * (THRESHOLDS["hp"] + THRESHOLDS["def"]):
                return None
            if move_type == MoveCategory.SPECIAL and enemy.base_stats[
                "hp"
            ] + enemy.base_stats["spd"] > 2.5 * (THRESHOLDS["hp"] + THRESHOLDS["spd"]):
                return None
            if move in battle.available_moves:
                possible_moves.append(move)
        chosen_move = None
        for move in possible_moves:
            if chosen_move is None or chosen_move.base_power < move.base_power:
                chosen_move = move
        return chosen_move

    def can_status(self, battle: Gen8Battle) -> bool:
        return self.get_status(battle) is not None

    @staticmethod
    def get_status(battle: Gen8Battle) -> Optional[Move]:
        mon = battle.active_pokemon
        enemy = battle.opponent_active_pokemon
        if enemy.status is None:
            return None
        chosen_move = None
        for move in mon.moves.values():
            if move.status is not None:
                status = move.status
                if status == Status.PSN or status == Status.TOX:
                    if (
                        PokemonType.STEEL in enemy.types
                        or PokemonType.POISON in enemy.types
                    ):
                        continue
                if move in battle.available_moves:
                    chosen_move = move
        return chosen_move

    def resist(self, battle: Gen8Battle) -> bool:
        return self.get_resisted_move(battle) is not None

    def get_resisted_move(self, battle: Gen8Battle) -> Optional[Move]:
        mon = battle.active_pokemon
        enemy = battle.opponent_active_pokemon
        tmp = self.learnset[enemy.species]["moves"]
        opponent_possible_moves = []
        for m in tmp:
            opponent_possible_moves.append(
                Move(m.lower().replace(" ", "").replace("-", "").replace("'", ""))
            )
        for move in opponent_possible_moves:
            if mon.damage_multiplier(move) >= 2.0 and move.base_power >= 40:
                return None
        chosen_move = None
        for move in mon.moves.values():
            if enemy.damage_multiplier(move) >= 1.0 and move.base_power >= 40:
                if (
                    chosen_move is None or chosen_move.base_power < move.base_power
                ) and move in battle.available_moves:
                    chosen_move = move
        return chosen_move

    def do_i_get_oneshot(self, battle: Gen8Battle) -> bool:
        pass
