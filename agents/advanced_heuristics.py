# Player with advanced heuristics
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.move import Move, DynamaxMove
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player
from typing import Awaitable, Dict, List, Optional, Union

from utils.get_smogon_data import MEGA_STONES


DAMAGE_REDUCTION_COEFFICIENT = 0.2
HIGH_STAT_THRESHOLD = 90
STRONG_MON_STATS_THRESHOLD = 500


class BattleInfo:
    def __init__(self):
        self.to_mega: Optional[Pokemon] = None
        self.z_move: Optional[tuple[Pokemon, Move]] = None
        self.to_dynamax: Optional[Pokemon] = None
        self.programmed_switch: Optional[Pokemon] = None

    def __str__(self):
        return (
            f"Pokemon to mega evolve: {self.to_mega.species if self.to_mega is not None else None} - "
            f"Pokemon and z-move to use: {self.z_move[0].species if self.z_move is not None else None}, "
            f"{self.z_move[1].id if self.z_move is not None else None} - "
            f"Pokemon to dynamax: {self.to_dynamax.species if self.to_dynamax is not None else None} - "
            f"Programmed switch: {self.programmed_switch.species if self.programmed_switch is not None else None}"
        )


class AdvancedHeuristics(Player):
    def __init__(self, *args, **kwargs):
        self._battle_info: Dict[str, BattleInfo] = {}
        super().__init__(*args, **kwargs)

    def choose_move(
        self, battle: AbstractBattle
    ) -> Union[BattleOrder, Awaitable[BattleOrder]]:
        if battle.battle_tag not in self._battle_info.keys():
            self.init_battle(battle)
        return self.get_next_order(battle, self._battle_info[battle.battle_tag])

    def get_next_order(
        self, battle: AbstractBattle, battle_info: BattleInfo
    ) -> BattleOrder:
        # Handle forced switching
        force_switch = battle.force_switch
        if isinstance(force_switch, list):
            force_switch = any(force_switch)
        if force_switch:
            if battle_info.programmed_switch is not None:
                switch = battle_info.programmed_switch
                battle_info.programmed_switch = None
                if switch in battle.available_switches:
                    return self.create_order(switch)
            switch = self.get_best_clean_switch(battle)
            if switch is None:
                return self.choose_random_move(battle)
            return self.create_order(switch)

        # Handle voluntary switching
        possible_switch = self.get_voluntary_switch(battle, battle_info)
        if possible_switch is not None:
            return self.create_order(possible_switch)

        # Handle Z Move
        if (
            battle.can_z_move
            and battle_info.z_move is not None
            and battle_info.z_move[0].species == battle.active_pokemon.species
        ):
            return self.create_order(battle_info.z_move[1], z_move=True)

        # Handle attacking move with dynamax and mega evolution
        best_move = self.get_best_attack_move(
            battle.active_pokemon,
            battle.opponent_active_pokemon,
            battle.available_moves,
        )
        if best_move is not None:
            if (
                battle.can_dynamax
                and battle_info.to_dynamax is not None
                and battle_info.to_dynamax.species == battle.active_pokemon.species
            ):
                return self.create_order(best_move, dynamax=True)
            elif (
                battle.can_mega_evolve
                and battle_info.to_mega is not None
                and battle_info.to_mega.species == battle.active_pokemon.species
            ):
                return self.create_order(best_move, mega=True)
            else:
                return self.create_order(best_move)
        return self.choose_random_move(battle)

    def get_voluntary_switch(
        self, battle: AbstractBattle, battle_info: BattleInfo
    ) -> Optional[Pokemon]:
        return None

    def get_best_clean_switch(self, battle: AbstractBattle) -> Optional[Pokemon]:
        switches = battle.available_switches
        if len(switches) == 0:
            return None
        return switches[0]

    def get_best_attack_move(
        self, mon: Pokemon, opponent: Pokemon, moves: List[Move]
    ) -> Optional[Move]:
        selected = None
        max_value = 0
        for move in moves:
            if mon.is_dynamaxed and not isinstance(move, DynamaxMove):
                move = move.dynamaxed
            move_value = (
                move.base_power * move.accuracy * self.damage_multiplier(move, opponent)
            )
            if move.type in mon.types:
                move_value *= 1.5
            if (
                opponent.base_stats["atk"] >= HIGH_STAT_THRESHOLD
                and move.category == MoveCategory.PHYSICAL
            ):
                move_value *= 1 - DAMAGE_REDUCTION_COEFFICIENT
            if (
                opponent.base_stats["spd"] >= HIGH_STAT_THRESHOLD
                and move.category == MoveCategory.SPECIAL
            ):
                move_value *= 1 - DAMAGE_REDUCTION_COEFFICIENT
            if move_value > max_value:
                selected = move
                max_value = move_value
        return selected

    def init_battle(self, battle: AbstractBattle):
        info = BattleInfo()

        # Who to dynamax?
        mon_value = 0
        for mon in battle.team.values():
            value = (
                mon.base_stats["atk"] + mon.base_stats["spa"] + mon.base_stats["spe"]
            )
            if value > mon_value:
                mon_value = value
                info.to_dynamax = mon

        # Who to mega evolve?
        mon_value = 0
        for mon in battle.team.values():
            if mon.item in MEGA_STONES:
                value = (
                    mon.base_stats["atk"]
                    + mon.base_stats["spa"]
                    + mon.base_stats["spe"]
                )
                if value > mon_value:
                    info.to_mega = mon
                    mon_value = value

        # What z-move to use?
        mon_value = 0
        for mon in battle.team.values():
            for move in mon.available_z_moves:
                move_value = move.z_move_power
                if move.category == MoveCategory.PHYSICAL:
                    move_value *= mon.base_stats["atk"] + mon.base_stats["spe"]
                elif move.category == MoveCategory.SPECIAL:
                    move_value *= mon.base_stats["spa"] + mon.base_stats["spe"]
                if move_value > mon_value:
                    info.z_move = (mon, move)
                    mon_value = move_value

        self._battle_info[battle.battle_tag] = info

    @staticmethod
    def is_strong(mon: Pokemon) -> bool:
        return sum(mon.base_stats.values()) >= STRONG_MON_STATS_THRESHOLD

    @staticmethod
    def damage_multiplier(move: Move, target: Pokemon) -> float:
        return target.damage_multiplier(move)

    def reset_battles(self) -> None:
        super().reset_battles()
        self._battle_info = {}
