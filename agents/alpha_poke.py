# Module containing production-level agents with neural networks
import os
import numpy as np
import tensorflow as tf

from abc import ABC
from gym.spaces import Space, Dict, Box
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.effect import Effect
from poke_env.environment.field import Field
from poke_env.environment.move import DynamaxMove, Move
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon, UNKNOWN_ITEM
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.side_condition import SideCondition, STACKABLE_CONDITIONS
from poke_env.environment.status import Status
from poke_env.environment.weather import Weather
from poke_env.player.baselines import (
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.player.openai_api import ObservationType
from poke_env.player.player import Player
from tensorflow.keras import activations, initializers, layers, losses, optimizers
from tf_agents.agents import TFAgent
from tf_agents.agents.dqn.dqn_agent import DqnAgent, DdqnAgent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.drivers.py_driver import PyDriver
from tf_agents.networks.nest_map import NestFlatten, NestMap
from tf_agents.networks.sequential import Sequential
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.specs import tensor_spec
from typing import Iterator, Union, List

from agents.base_classes.dqn_player import DQNPlayer
from agents.seba import Seba
from utils.get_smogon_data import get_abilities, get_items

STATS = {
    "hp": 0,
    "atk": 1,
    "def": 2,
    "spa": 3,
    "spd": 4,
    "spe": 5,
    "accuracy": 6,
    "evasion": 7,
}

BOOSTS_MULTIPLIERS = [0.25, 0.28, 0.33, 0.4, 0.5, 0.66, 1, 1.5, 2, 2.5, 3, 3.5, 4]


INFINITE_WEATHER = [Weather.DELTASTREAM, Weather.PRIMORDIALSEA, Weather.DESOLATELAND]

ABILITIES = get_abilities(8)

ITEMS = get_items()


class _BattlefieldEmbedding:
    @staticmethod
    def embed_battlefield(battle: AbstractBattle):
        dynamax_turns = np.full(2, -1, dtype=int)
        if battle.dynamax_turns_left is not None:
            dynamax_turns[0] = battle.dynamax_turns_left
        if battle.opponent_dynamax_turns_left is not None:
            dynamax_turns[1] = battle.opponent_dynamax_turns_left

        boolean_flags = np.full(6, 0, dtype=int)
        if battle.can_mega_evolve:
            boolean_flags[0] = 1
        if battle.can_z_move:
            boolean_flags[1] = 1
        if battle.can_dynamax:
            boolean_flags[2] = 1
        if battle.opponent_can_dynamax:
            boolean_flags[3] = 1
        if battle.maybe_trapped:
            boolean_flags[4] = 1
        if battle.force_switch:
            boolean_flags[5] = 1

        return {
            "dynamax_turns": dynamax_turns,
            "boolean_flags": boolean_flags,
            "fields": _FieldEmbedding.embed_field(battle),
            "side_conditions": _SideConditionEmbedding.embed_side_conditions(battle),
            "weather": _WeatherEmbedding.embed_weather(battle),
        }

    @staticmethod
    def get_embedding():
        dynamax_turns_low = [-1, -1]
        dynamax_turns_high = [3, 3]
        dynamax_turns = Box(
            low=np.array(dynamax_turns_low, dtype=int),
            high=np.array(dynamax_turns_high, dtype=int),
            dtype=int,
        )
        boolean_flags = Box(low=0, high=1, shape=(6,), dtype=int)
        return Dict(
            {
                "dynamax_turns": dynamax_turns,
                "boolean_flags": boolean_flags,
                "fields": _FieldEmbedding.get_embedding(),
                "side_conditions": _SideConditionEmbedding.get_embedding(),
                "weather": _WeatherEmbedding.get_embedding(),
            }
        )


class _ActivePokemonEmbedding:
    @staticmethod
    def embed_pokemon(mon: Pokemon, battle: AbstractBattle):
        current_hp_fraction = np.full(1, -1.0, dtype=np.float64)
        if mon is not None:
            current_hp_fraction[0] = mon.current_hp_fraction
        available_moves = [None, None, None, None]
        for i, move in enumerate(battle.available_moves):
            available_moves[i] = move
        return {
            "current_hp_fraction": current_hp_fraction,
            "base_stats": _BaseStatsEmbedding.embed_stats(mon),
            "type": _TypeEmbedding.embed_type(mon),
            "boosts": _MonBoostsEmbedding.embed_boosts(mon),
            "status": _StatusEmbedding.embed_status(mon),
            "effects": _EffectsEmbedding.embed_effects(mon),
            "move_1": _MoveEmbedding.embed_move(
                available_moves[0], mon, battle.opponent_active_pokemon
            ),
            "move_2": _MoveEmbedding.embed_move(
                available_moves[1], mon, battle.opponent_active_pokemon
            ),
            "move_3": _MoveEmbedding.embed_move(
                available_moves[2], mon, battle.opponent_active_pokemon
            ),
            "move_4": _MoveEmbedding.embed_move(
                available_moves[3], mon, battle.opponent_active_pokemon
            ),
        }

    @staticmethod
    def get_embedding() -> Space:
        current_hp_fraction_space = Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float64
        )
        return Dict(
            {
                "current_hp_fraction": current_hp_fraction_space,
                "base_stats": _BaseStatsEmbedding.get_embedding(),
                "type": _TypeEmbedding.get_embedding(),
                "boosts": _MonBoostsEmbedding.get_embedding(),
                "status": _StatusEmbedding.get_embedding(),
                "effects": _EffectsEmbedding.get_embedding(),
                "move_1": _MoveEmbedding.get_embedding(),
                "move_2": _MoveEmbedding.get_embedding(),
                "move_3": _MoveEmbedding.get_embedding(),
                "move_4": _MoveEmbedding.get_embedding(),
            }
        )


class _PokemonEmbedding:
    @staticmethod
    def embed_pokemon(mon: Pokemon, battle: AbstractBattle):
        moves = []
        current_hp_fraction = np.full(1, -1.0, dtype=np.float64)
        if mon is not None:
            current_hp_fraction[0] = mon.current_hp_fraction
            moves = list(mon.moves.values())
        while len(moves) < 4:
            moves.append(None)
        return {
            "current_hp_fraction": current_hp_fraction,
            "base_stats": _BaseStatsEmbedding.embed_stats(mon),
            "type": _TypeEmbedding.embed_type(mon),
            "status": _StatusEmbedding.embed_status(mon),
            "move_1": _MoveEmbedding.embed_move(
                moves[0], mon, battle.opponent_active_pokemon
            ),
            "move_2": _MoveEmbedding.embed_move(
                moves[1], mon, battle.opponent_active_pokemon
            ),
            "move_3": _MoveEmbedding.embed_move(
                moves[2], mon, battle.opponent_active_pokemon
            ),
            "move_4": _MoveEmbedding.embed_move(
                moves[3], mon, battle.opponent_active_pokemon
            ),
        }

    @staticmethod
    def get_embedding() -> Space:
        current_hp_fraction_space = Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float64
        )
        return Dict(
            {
                "current_hp_fraction": current_hp_fraction_space,
                "base_stats": _BaseStatsEmbedding.get_embedding(),
                "type": _TypeEmbedding.get_embedding(),
                "status": _StatusEmbedding.get_embedding(),
                "move_1": _MoveEmbedding.get_embedding(),
                "move_2": _MoveEmbedding.get_embedding(),
                "move_3": _MoveEmbedding.get_embedding(),
                "move_4": _MoveEmbedding.get_embedding(),
            }
        )


class _EnemyActivePokemonEmbedding:
    @staticmethod
    def embed_pokemon(mon: Pokemon, battle: AbstractBattle):
        moves = []
        current_hp_fraction = np.full(1, -1.0, dtype=np.float64)
        if mon is not None:
            current_hp_fraction[0] = mon.current_hp_fraction
            moves = list(mon.moves.values())
            for move in moves:
                if isinstance(move, DynamaxMove) != mon.is_dynamaxed:
                    moves.remove(move)
        while len(moves) < 4:
            moves.append(None)
        return {
            "current_hp_fraction": current_hp_fraction,
            "base_stats": _BaseStatsEmbedding.embed_stats(mon),
            "type": _TypeEmbedding.embed_type(mon),
            "status": _StatusEmbedding.embed_status(mon),
            "boosts": _MonBoostsEmbedding.embed_boosts(mon),
            "move_1": _MoveEmbedding.embed_move(moves[0], mon, battle.active_pokemon),
            "move_2": _MoveEmbedding.embed_move(moves[1], mon, battle.active_pokemon),
            "move_3": _MoveEmbedding.embed_move(moves[2], mon, battle.active_pokemon),
            "move_4": _MoveEmbedding.embed_move(moves[3], mon, battle.active_pokemon),
        }

    @staticmethod
    def get_embedding() -> Space:
        current_hp_fraction_space = Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float64
        )
        return Dict(
            {
                "current_hp_fraction": current_hp_fraction_space,
                "base_stats": _BaseStatsEmbedding.get_embedding(),
                "type": _TypeEmbedding.get_embedding(),
                "status": _StatusEmbedding.get_embedding(),
                "boosts": _MonBoostsEmbedding.get_embedding(),
                "move_1": _MoveEmbedding.get_embedding(),
                "move_2": _MoveEmbedding.get_embedding(),
                "move_3": _MoveEmbedding.get_embedding(),
                "move_4": _MoveEmbedding.get_embedding(),
            }
        )


class _EnemyPokemonEmbedding:
    @staticmethod
    def embed_pokemon(mon: Pokemon, battle: AbstractBattle):
        moves = []
        current_hp_fraction = np.full(1, -1.0, dtype=np.float64)
        if mon is not None:
            current_hp_fraction[0] = mon.current_hp_fraction
            moves = list(mon.moves.values())
        while len(moves) < 4:
            moves.append(None)
        return {
            "current_hp_fraction": current_hp_fraction,
            "base_stats": _BaseStatsEmbedding.embed_stats(mon),
            "type": _TypeEmbedding.embed_type(mon),
            "status": _StatusEmbedding.embed_status(mon),
            "move_1": _MoveEmbedding.embed_move(moves[0], mon, battle.active_pokemon),
            "move_2": _MoveEmbedding.embed_move(moves[1], mon, battle.active_pokemon),
            "move_3": _MoveEmbedding.embed_move(moves[2], mon, battle.active_pokemon),
            "move_4": _MoveEmbedding.embed_move(moves[3], mon, battle.active_pokemon),
        }

    @staticmethod
    def get_embedding() -> Space:
        current_hp_fraction_space = Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float64
        )
        return Dict(
            {
                "current_hp_fraction": current_hp_fraction_space,
                "base_stats": _BaseStatsEmbedding.get_embedding(),
                "type": _TypeEmbedding.get_embedding(),
                "status": _StatusEmbedding.get_embedding(),
                "move_1": _MoveEmbedding.get_embedding(),
                "move_2": _MoveEmbedding.get_embedding(),
                "move_3": _MoveEmbedding.get_embedding(),
                "move_4": _MoveEmbedding.get_embedding(),
            }
        )


# Array embedding of boosts
class _MonBoostsEmbedding:
    @staticmethod
    def embed_boosts(mon: Pokemon):
        if mon is None:
            return np.full(len(STATS) - 1, -1.0, dtype=np.float64)
        boosts = np.full(len(STATS) - 1, 1.0, dtype=np.float64)
        mon_boosts = mon.boosts
        for boost, boost_value in mon_boosts.items():
            stat_index = STATS[boost] - 1
            boost_multiplier = BOOSTS_MULTIPLIERS[boost_value + 6]
            boosts[stat_index] = boost_multiplier
        return boosts

    @staticmethod
    def get_embedding():
        low_bound = [-1.0 for _ in range(len(STATS) - 1)]
        high_bound = [4.0 for _ in range(len(STATS) - 1)]
        return Box(
            low=np.array(low_bound, dtype=np.float64),
            high=np.array(high_bound, dtype=np.float64),
            dtype=np.float64,
        )


# Array embedding of base stats
class _BaseStatsEmbedding:
    @staticmethod
    def embed_stats(mon: Pokemon):
        if mon is None:
            return np.full(6, -1.0, dtype=np.float64)
        stats = np.full(6, 0.0, dtype=np.float64)
        stats[0] = mon.base_stats["hp"] / 255
        stats[1] = mon.base_stats["atk"] / 255
        stats[2] = mon.base_stats["def"] / 255
        stats[3] = mon.base_stats["spa"] / 255
        stats[4] = mon.base_stats["spd"] / 255
        stats[5] = mon.base_stats["spe"] / 255
        return stats

    @staticmethod
    def get_embedding() -> Space:
        low_bound = [-1.0 for _ in range(6)]
        high_bound = [1.0 for _ in range(6)]
        return Box(
            low=np.array(low_bound, dtype=np.float64),
            high=np.array(high_bound, dtype=np.float64),
            dtype=np.float64,
        )


# Dict embedding of a move.
class _MoveEmbedding:
    @staticmethod
    def embed_move(move: Move, mon: Pokemon, opponent: Pokemon):
        if move is None:
            base_power = -1.0
            accuracy = -1.0
            pps = -1.0
            drain = -1.0
            heal = -1.0
            recoil = -1.0
            damage_multiplier = -1.0
            min_hits = -1
            max_hits = -1
            mean_hits = -1.0
            crit_ratio = -1
            priority = -8
            damage = -1
        else:
            base_power = move.base_power / 100
            accuracy = move.accuracy
            pps = move.current_pp / move.max_pp
            drain = move.drain
            heal = move.heal
            recoil = move.recoil
            damage_multiplier = opponent.damage_multiplier(move)
            min_hits, max_hits = move.n_hit
            mean_hits = move.expected_hits
            crit_ratio = move.crit_ratio
            priority = move.priority
            damage = move.damage
            if damage == "level":
                damage = mon.level
        float_move_info = np.array(
            [
                base_power,
                accuracy,
                pps,
                drain,
                heal,
                mean_hits,
                recoil,
                damage_multiplier,
            ],
            dtype=np.float64,
        )
        int_move_info = np.array(
            [min_hits, max_hits, crit_ratio, priority, damage], dtype=int
        )
        return {
            "float_move_info": float_move_info,
            "int_move_info": int_move_info,
            "move_category": _MoveCategoryEmbedding.embed_category(move),
            "move_type": _TypeEmbedding.embed_type(move),
            "move_flags": _MoveFlagsEmbedding.embed_move_flags(move, opponent),
            "move_status": _MoveStatusEmbedding.embed_move_status(move),
            "boosts": _BoostsEmbedding.embed_boosts(move),
            "self_boosts": _SelfBoostsEmbedding.embed_self_boosts(move),
        }

    @staticmethod
    def get_embedding() -> Space:
        float_info_low_bound = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        float_info_high_bound = [4.0, 1.0, 1.0, 1.0, 1.0, 5.23, 1.0, 4.0]
        #                                                 ^^^^
        # 5.23 is the expected hit number for triple kick and triple axel
        #
        float_info_space = Box(
            low=np.array(float_info_low_bound, dtype=np.float64),
            high=np.array(float_info_high_bound, dtype=np.float64),
            dtype=np.float64,
        )
        int_info_low_bound = [-1, -1, -1, -8, -1]
        int_info_high_bound = [5, 5, 6, 5, 100]
        int_info_space = Box(
            low=np.array(int_info_low_bound, dtype=int),
            high=np.array(int_info_high_bound, dtype=int),
            dtype=int,
        )
        return Dict(
            {
                "float_move_info": float_info_space,
                "int_move_info": int_info_space,
                "move_category": _MoveCategoryEmbedding.get_embedding(),
                "move_type": _TypeEmbedding.get_embedding(),
                "move_flags": _MoveFlagsEmbedding.get_embedding(),
                "move_status": _MoveStatusEmbedding.get_embedding(),
                "boosts": _BoostsEmbedding.get_embedding(),
                "self_boosts": _SelfBoostsEmbedding.get_embedding(),
            }
        )


# Array of int flags for the move embedding
class _MoveFlagsEmbedding:
    @staticmethod
    def embed_move_flags(move: Move, opponent: Pokemon):
        if move is None:
            return np.full(6, -1, dtype=int)
        flags = np.full(6, 0, dtype=int)
        if move.can_z_move:
            flags[0] = 1
        if move.thaws_target:
            flags[1] = 1
        if move.stalling_move:
            flags[2] = 1
        if move.ignore_immunity and opponent is not None:
            if isinstance(move.ignore_immunity, bool):
                flags[3] = 1
            else:
                for t in opponent.types:
                    if t in move.ignore_immunity:
                        flags[3] = 1
        if move.force_switch:
            flags[4] = 1
        if move.breaks_protect:
            flags[5] = 1
        return flags

    @staticmethod
    def get_embedding() -> Space:
        low_bound = [-1 for _ in range(6)]
        high_bound = [1 for _ in range(6)]
        return Box(
            low=np.array(low_bound, dtype=int),
            high=np.array(high_bound, dtype=int),
            dtype=int,
        )


# One hot encoding for the move category
class _MoveCategoryEmbedding:
    @staticmethod
    def embed_category(move: Move):
        if move is None:
            return np.full(len(MoveCategory), -1, dtype=int)
        category = np.full(len(MoveCategory), 0, dtype=int)
        category[move.category.value - 1] = 1
        return category

    @staticmethod
    def get_embedding() -> Space:
        low_bound = [-1 for _ in range(len(MoveCategory))]
        high_bound = [1 for _ in range(len(MoveCategory))]
        return Box(
            low=np.array(low_bound, dtype=int),
            high=np.array(high_bound, dtype=int),
            dtype=int,
        )


# Two arrays. One with the status, the other with the chance of it happening.
class _MoveStatusEmbedding:
    @staticmethod
    def embed_move_status(move: Move):
        if move is None:
            status = np.full(len(Status), -1, dtype=int)
            chance = np.full(len(Status), -1, dtype=np.float64)
        else:
            status = np.full(len(Status), 0, dtype=int)
            chance = np.full(len(Status), 0, dtype=np.float64)
            if move.status is not None:
                status[move.status.value - 1] = 1
                chance[move.status.value - 1] = 1.0
            else:
                secondary = move.secondary
                for d in secondary:
                    if "status" in d.keys():
                        secondary_chance = d["chance"] / 100
                        secondary_status = getattr(Status, d["status"].upper())
                        status[secondary_status.value - 1] = 1
                        chance[secondary_status.value - 1] = secondary_chance
        return {"status": status, "chances": chance}

    @staticmethod
    def get_embedding() -> Space:
        status_low_bound = [-1 for _ in range(len(Status))]
        status_high_bound = [1 for _ in range(len(Status))]
        status_space = Box(
            low=np.array(status_low_bound, dtype=int),
            high=np.array(status_high_bound, dtype=int),
            dtype=int,
        )
        chance_low_bound = [-1.0 for _ in range(len(Status))]
        chance_high_bound = [1.0 for _ in range(len(Status))]
        chance_space = Box(
            low=np.array(chance_low_bound, dtype=np.float64),
            high=np.array(chance_high_bound, dtype=np.float64),
            dtype=np.float64,
        )
        return Dict({"status": status_space, "chances": chance_space})


# Two arrays. One with the boost, the other with the chance of it happening.
class _BoostsEmbedding:
    @staticmethod
    def embed_boosts(move: Move):
        if move is None:
            boosts = np.full(7, -7, dtype=int)
            chance = np.full(7, -1, dtype=np.float64)
        else:
            boosts = np.full(7, 0, dtype=int)
            chance = np.full(7, 0, dtype=np.float64)
            secondary = move.secondary
            move_boosts = {}
            if move.target != "self" and move.boosts is not None:
                move_boosts.update(move.boosts)
            secondary_boosts = {}
            for d in secondary:
                if "boosts" in d.keys():
                    secondary_chance = d["chance"] / 100
                    for key, value in d["boosts"].items():
                        secondary_boosts[key] = (value, secondary_chance)
            for key, value in move_boosts.items():
                boosts[STATS[key] - 1] = value
                chance[STATS[key] - 1] = 1.0
            for key, value in secondary_boosts.items():
                boosts[STATS[key] - 1] = value[0]
                chance[STATS[key] - 1] = value[1]
        return {"boosts": boosts, "chances": chance}

    @staticmethod
    def get_embedding() -> Space:
        boosts_low_bound = [-7 for _ in range(7)]
        boosts_high_bound = [6 for _ in range(7)]
        boosts_space = Box(
            low=np.array(boosts_low_bound, dtype=int),
            high=np.array(boosts_high_bound, dtype=int),
            dtype=int,
        )
        chance_low_bound = [-1.0 for _ in range(7)]
        chance_high_bound = [1.0 for _ in range(7)]
        chance_space = Box(
            low=np.array(chance_low_bound, dtype=np.float64),
            high=np.array(chance_high_bound, dtype=np.float64),
            dtype=np.float64,
        )
        return Dict({"boosts": boosts_space, "chances": chance_space})


# Two arrays. One with the boost, the other with the chance of it happening.
class _SelfBoostsEmbedding:
    @staticmethod
    def embed_self_boosts(move: Move):
        if move is None:
            self_boosts = np.full(7, -7, dtype=int)
            chance = np.full(7, -1, dtype=np.float64)
        else:
            self_boosts = np.full(7, 0, dtype=int)
            chance = np.full(7, 0, dtype=np.float64)
            secondary = move.secondary
            boosts = {}
            if move.self_boost is not None:
                boosts.update(move.self_boost)
            if move.target == "self" and move.boosts is not None:
                if move.self_boost is not None:
                    raise RuntimeError(
                        "Did not expect self_boosts and boosts to be active at the same time."
                    )
                boosts.update(move.boosts)
            secondary_boosts = {}
            for d in secondary:
                if "self" in d.keys():
                    data = d["self"]
                    boost_chance = d["chance"] / 100
                    if len(data) == 1 and list(data.keys()) == ["boosts"]:
                        for key, value in data["boosts"].items():
                            secondary_boosts[key] = (value, boost_chance)
            for key, value in boosts.items():
                self_boosts[STATS[key] - 1] = value
                chance[STATS[key] - 1] = 1.0
            for key, value in secondary_boosts.items():
                self_boosts[STATS[key] - 1] = value[0]
                chance[STATS[key] - 1] = value[1]
        return {"boosts": self_boosts, "chances": chance}

    @staticmethod
    def get_embedding() -> Space:
        self_boosts_low_bound = [-7 for _ in range(7)]
        self_boosts_high_bound = [6 for _ in range(7)]
        self_boosts_space = Box(
            low=np.array(self_boosts_low_bound, dtype=int),
            high=np.array(self_boosts_high_bound, dtype=int),
            dtype=int,
        )
        chance_low_bound = [-1.0 for _ in range(7)]
        chance_high_bound = [1.0 for _ in range(7)]
        chance_space = Box(
            low=np.array(chance_low_bound, dtype=np.float64),
            high=np.array(chance_high_bound, dtype=np.float64),
            dtype=np.float64,
        )
        return Dict({"boosts": self_boosts_space, "chances": chance_space})


# One hot encoding for move and pokémon types
class _TypeEmbedding:
    @staticmethod
    def embed_type(mon_or_move: Union[Pokemon, Move]):
        if mon_or_move is None:
            return np.full(len(PokemonType), -1, dtype=int)
        types = np.full(len(PokemonType), 0, dtype=int)
        if isinstance(mon_or_move, Move):
            battle_types = [mon_or_move.type]
        elif isinstance(mon_or_move, Pokemon):
            battle_types = mon_or_move.types
        else:
            raise RuntimeError(f"Expected Move or Pokemon, got {type(mon_or_move)}.")
        for mon_type in battle_types:
            if mon_type is not None:
                types[mon_type.value - 1] = 1
        return types

    @staticmethod
    def get_embedding() -> Space:
        low_bound = [-1 for _ in range(len(PokemonType))]
        high_bound = [1 for _ in range(len(PokemonType))]
        return Box(
            low=np.array(low_bound, dtype=int),
            high=np.array(high_bound, dtype=int),
            dtype=int,
        )


# One hot encoding for Pokémon items.
class _ItemEmbedding:
    @staticmethod
    def embed_item(mon: Pokemon):
        if mon.item is None or mon.item == UNKNOWN_ITEM:
            return np.full(len(ITEMS), -1, dtype=int)
        battle_item = mon.item
        items = np.full(len(ITEMS), 0, dtype=int)
        items[getattr(ITEMS, battle_item).value - 1] = 1
        return items

    @staticmethod
    def get_embedding() -> Space:
        low_bound = [-1 for _ in range(len(ITEMS))]
        high_bound = [1 for _ in range(len(ITEMS))]
        return Box(
            low=np.array(low_bound, dtype=int),
            high=np.array(high_bound, dtype=int),
            dtype=int,
        )


# One hot encoding for the Pokémon abilities.
class _AbilityEmbedding:
    @staticmethod
    def embed_ability(mon: Pokemon):
        if mon is None:
            return np.full(len(ABILITIES), -1, dtype=int)
        battle_abilities = np.full(len(ABILITIES), 0, dtype=int)
        if mon.ability is None:
            possible_abilities = mon.possible_abilities
            if len(possible_abilities) == 1:
                for ability in possible_abilities:
                    battle_abilities[getattr(ABILITIES, ability).value - 1] = 2
            else:
                for ability in possible_abilities:
                    battle_abilities[getattr(ABILITIES, ability).value - 1] = 1
            return battle_abilities
        battle_abilities[getattr(ABILITIES, mon.ability).value - 1] = 2
        return battle_abilities

    @staticmethod
    def get_embedding() -> Space:
        low_bound = [-1 for _ in range(len(ABILITIES))]
        high_bound = [2 for _ in range(len(ABILITIES))]
        return Box(
            low=np.array(low_bound, dtype=int),
            high=np.array(high_bound, dtype=int),
            dtype=int,
        )


# One hot encoding for the weather.
class _WeatherEmbedding:
    @staticmethod
    def embed_weather(battle: AbstractBattle):
        current_turn = battle.turn
        weather = battle.weather
        weathers = np.full(len(Weather), -1, dtype=int)
        for w in INFINITE_WEATHER:
            weathers[w.value - 1] = 0
        for w, value in weather.items():
            if w in INFINITE_WEATHER:
                weathers[w.value - 1] = 1
            else:
                weathers[w.value - 1] = current_turn - value
        return weathers

    @staticmethod
    def get_embedding() -> Space:
        low_bound = [-1 for _ in range(len(Weather))]
        high_bound = [6 for _ in range(len(Weather))]
        for w in INFINITE_WEATHER:
            high_bound[w.value - 1] = 1
            low_bound[w.value - 1] = 0
        return Box(
            low=np.array(low_bound, dtype=int),
            high=np.array(high_bound, dtype=int),
            dtype=int,
        )


# One hot encoding for the Pokémon statuses.
class _StatusEmbedding:
    @staticmethod
    def embed_status(mon: Pokemon):
        if mon is not None:
            status = mon.status
            statuses = np.full(len(Status), 0, dtype=int)
            if status is not None:
                statuses[status.value - 1] = 1
        else:
            statuses = np.full(len(Status), -1, dtype=int)
        return statuses

    @staticmethod
    def get_embedding() -> Space:
        low_bound = [-1 for _ in range(len(Status))]
        high_bound = [1 for _ in range(len(Status))]
        return Box(
            low=np.array(low_bound, dtype=int),
            high=np.array(high_bound, dtype=int),
            dtype=int,
        )


# One hot encoding for the Pokémon effects.
class _EffectsEmbedding:
    @staticmethod
    def embed_effects(mon: Pokemon):
        battle_effects = {}
        if mon is not None:
            battle_effects = mon.effects
        effects = np.full(len(Effect), -1, dtype=int)
        for effect, counter in battle_effects.items():
            effects[effect.value - 1] = counter
        return effects

    @staticmethod
    def get_embedding() -> Space:
        low_bound = [-1 for _ in range(len(Effect))]
        high_bound = [6 for _ in range(len(Effect))]
        return Box(
            low=np.array(low_bound, dtype=int),
            high=np.array(high_bound, dtype=int),
            dtype=int,
        )


# One hot encoding for the side conditions.
class _SideConditionEmbedding:
    @staticmethod
    def embed_side_conditions(battle: AbstractBattle):
        current_turn = battle.turn
        battle_side_conditions = battle.side_conditions
        opponent_battle_side_conditions = battle.opponent_side_conditions
        side_conditions = np.full(len(SideCondition), -1, dtype=int)
        opponent_side_conditions = np.full(len(SideCondition), -1, dtype=int)
        side_conditions[SideCondition.STEALTH_ROCK.value - 1] = 0
        opponent_side_conditions[SideCondition.STEALTH_ROCK.value - 1] = 0
        for condition in STACKABLE_CONDITIONS.keys():
            side_conditions[condition.value - 1] = 0
            opponent_side_conditions[condition.value - 1] = 0
        for condition, value in battle_side_conditions.items():
            if condition in STACKABLE_CONDITIONS.keys():
                side_conditions[condition.value - 1] = value
            elif condition == SideCondition.STEALTH_ROCK:
                side_conditions[condition.value - 1] = 1
            else:
                side_conditions[condition.value - 1] = current_turn - value
        for condition, value in opponent_battle_side_conditions.items():
            if condition in STACKABLE_CONDITIONS.keys():
                opponent_side_conditions[condition.value - 1] = value
            elif condition == SideCondition.STEALTH_ROCK:
                opponent_side_conditions[condition.value - 1] = 1
            else:
                opponent_side_conditions[condition.value - 1] = current_turn - value
        return {
            "player_conditions": side_conditions,
            "opponent_conditions": opponent_side_conditions,
        }

    @staticmethod
    def get_embedding() -> Space:
        low_bound = [-1 for _ in range(len(SideCondition))]
        high_bound = [6 for _ in range(len(SideCondition))]
        low_bound[SideCondition.STEALTH_ROCK.value] = 0
        high_bound[SideCondition.STEALTH_ROCK.value] = 1
        for condition in STACKABLE_CONDITIONS.keys():
            low_bound[condition.value - 1] = 0
            high_bound[condition.value - 1] = STACKABLE_CONDITIONS[condition]
        bound_box = Box(
            low=np.array(low_bound, dtype=int),
            high=np.array(high_bound, dtype=int),
            dtype=int,
        )
        return Dict({"player_conditions": bound_box, "opponent_conditions": bound_box})


# One hot encoding for the fields.
class _FieldEmbedding:
    @staticmethod
    def embed_field(battle: AbstractBattle):
        current_turn = battle.turn
        fields = np.full(len(Field), -1, dtype=int)
        battle_fields = battle.fields
        for field, value in battle_fields.items():
            fields[field.value - 1] = current_turn - value
        return fields

    @staticmethod
    def get_embedding() -> Space:
        low_bound = [-1 for _ in range(len(Field))]
        high_bound = [6 for _ in range(len(Field))]
        return Box(
            low=np.array(low_bound, dtype=int),
            high=np.array(high_bound, dtype=int),
            dtype=int,
        )


class AlphaPokeSingleEmbedded(DQNPlayer, ABC):
    def __init__(self, log_interval=1000, eval_interval=10_000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.format_is_doubles:
            raise NotImplementedError("Double battles are not supported by this class")
        self.log_int = log_interval
        self.eval_int = eval_interval

    def fainted_value(self) -> float:
        return 0.0

    def hp_value(self) -> float:
        return 0.0

    def number_of_pokemons(self) -> int:
        return 6

    def starting_value(self) -> float:
        return 0.0

    def status_value(self) -> float:
        return 0.0

    def victory_value(self) -> float:
        return 1.0

    def calc_reward(
        self, last_battle: AbstractBattle, current_battle: AbstractBattle
    ) -> float:
        return self.reward_computing_helper(
            current_battle,
            fainted_value=self.fainted_value(),
            hp_value=self.hp_value(),
            number_of_pokemons=self.number_of_pokemons(),
            starting_value=self.starting_value(),
            status_value=self.status_value(),
            victory_value=self.victory_value(),
        )

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        non_active_player_mons = list(battle.team.values())
        non_active_opponent_mons = list(battle.opponent_team.values())
        non_active_player_mons.remove(battle.active_pokemon)
        non_active_opponent_mons.remove(battle.opponent_active_pokemon)
        while len(non_active_player_mons) < 5:
            non_active_player_mons.append(None)
        while len(non_active_opponent_mons) < 5:
            non_active_opponent_mons.append(None)
        return {
            "battlefield": _BattlefieldEmbedding.embed_battlefield(battle),
            "active_mon": _ActivePokemonEmbedding.embed_pokemon(
                battle.active_pokemon, battle
            ),
            "player_mon_1": _PokemonEmbedding.embed_pokemon(
                non_active_player_mons[0], battle
            ),
            "player_mon_2": _PokemonEmbedding.embed_pokemon(
                non_active_player_mons[1], battle
            ),
            "player_mon_3": _PokemonEmbedding.embed_pokemon(
                non_active_player_mons[2], battle
            ),
            "player_mon_4": _PokemonEmbedding.embed_pokemon(
                non_active_player_mons[3], battle
            ),
            "player_mon_5": _PokemonEmbedding.embed_pokemon(
                non_active_player_mons[4], battle
            ),
            "opponent_active_mon": _EnemyActivePokemonEmbedding.embed_pokemon(
                battle.opponent_active_pokemon, battle
            ),
            "opponent_mon_1": _EnemyPokemonEmbedding.embed_pokemon(
                non_active_opponent_mons[0], battle
            ),
            "opponent_mon_2": _EnemyPokemonEmbedding.embed_pokemon(
                non_active_opponent_mons[1], battle
            ),
            "opponent_mon_3": _EnemyPokemonEmbedding.embed_pokemon(
                non_active_opponent_mons[2], battle
            ),
            "opponent_mon_4": _EnemyPokemonEmbedding.embed_pokemon(
                non_active_opponent_mons[3], battle
            ),
            "opponent_mon_5": _EnemyPokemonEmbedding.embed_pokemon(
                non_active_opponent_mons[4], battle
            ),
        }

    @property
    def embedding(self) -> Space:
        return Dict(
            {
                "battlefield": _BattlefieldEmbedding.get_embedding(),
                "active_mon": _ActivePokemonEmbedding.get_embedding(),
                "player_mon_1": _PokemonEmbedding.get_embedding(),
                "player_mon_2": _PokemonEmbedding.get_embedding(),
                "player_mon_3": _PokemonEmbedding.get_embedding(),
                "player_mon_4": _PokemonEmbedding.get_embedding(),
                "player_mon_5": _PokemonEmbedding.get_embedding(),
                "opponent_active_mon": _EnemyActivePokemonEmbedding.get_embedding(),
                "opponent_mon_1": _EnemyPokemonEmbedding.get_embedding(),
                "opponent_mon_2": _EnemyPokemonEmbedding.get_embedding(),
                "opponent_mon_3": _EnemyPokemonEmbedding.get_embedding(),
                "opponent_mon_4": _EnemyPokemonEmbedding.get_embedding(),
                "opponent_mon_5": _EnemyPokemonEmbedding.get_embedding(),
            }
        )

    @property
    def opponents(self) -> Union[Player, str, List[Player], List[str]]:
        opponents_classes = [
            RandomPlayer,
            MaxBasePowerPlayer,
            SimpleHeuristicsPlayer,
            Seba,
        ]
        return [cls(battle_format=self.battle_format) for cls in opponents_classes]

    @property
    def log_interval(self) -> int:
        return self.log_int

    @property
    def eval_interval(self) -> int:
        return self.eval_int

    def eval_function(self, step):
        num_challenges = 50

        opponent = RandomPlayer(
            battle_format=self.battle_format, max_concurrent_battles=1
        )
        eval_env, agent = self.create_evaluation_env(active=True, opponents=[opponent])
        policy = self.agent.policy

        total_return = 0.0
        for _ in range(num_challenges):
            time_step = eval_env.reset()
            episode_return = 0.0
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step)
                episode_return += time_step.reward
            total_return += episode_return

        evaluation = (total_return / num_challenges).numpy()[0]

        if "evaluations" not in self.evaluations.keys():
            self.evaluations["evaluations"] = [[], []]
        self.evaluations["evaluations"][0].append(step)
        self.evaluations["evaluations"][1].append(evaluation)
        print(f"step: {step} - Average return: {evaluation}")
        eval_env.close()

    def log_function(self, step, loss_info: LossInfo):
        if "losses" not in self.evaluations.keys():
            self.evaluations["losses"] = [[], []]
        self.evaluations["losses"][0].append(step)
        self.evaluations["losses"][1].append(loss_info.loss)
        print(f"step: {step} - Loss: {loss_info.loss}")

    def save_training_data(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(save_dir + "/training_losses.csv", "w") as training_file:
            training_file.write("step;loss\n")
            for evaluation in self.evaluations["losses"]:
                training_file.write(f"{evaluation[0]};{evaluation[1]}\n")
        with open(save_dir + "/training_returns.csv", "w") as training_file:
            training_file.write("step;avg_returns\n")
            for evaluation in self.evaluations["evaluations"]:
                training_file.write(f"{evaluation[0]};{evaluation[1]}\n")


class AlphaPokeSingleDQN(AlphaPokeSingleEmbedded):
    def get_agent(self) -> TFAgent:
        action_tensor_spec = tensor_spec.from_spec(self.environment.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        obs_spec = tensor_spec.from_spec(self.environment.observation_spec())

        q_net = Sequential(self.get_network_layers(obs_spec, num_actions))
        optimizer = self.get_optimizer()

        train_step_counter = tf.Variable(0)

        agent = self.create_agent(q_net, optimizer, train_step_counter)
        return agent

    @staticmethod
    def get_network_layers(obs_spec, num_actions):
        dict_list = obs_spec.copy()
        to_see = [dict_list]
        for d in to_see:
            for key, value in d.items():
                if isinstance(value, dict):
                    d[key] = value.copy()
                    to_see.append(d[key])
                else:
                    d[key] = layers.Dense(
                        value.shape[0],
                        activation=activations.elu,
                        kernel_initializer=initializers.VarianceScaling(
                            scale=1.0, mode="fan_in", distribution="truncated_normal"
                        ),
                        use_bias=True,
                    )
        layer_list = [
            NestMap(dict_list, input_spec=obs_spec),
            NestFlatten(),
            layers.Concatenate(),
            layers.Dense(
                2048,
                activation=activations.elu,
                kernel_initializer=initializers.VarianceScaling(
                    scale=1.0, mode="fan_in", distribution="truncated_normal"
                ),
                use_bias=True,
            ),
            layers.Dense(
                1024,
                activation=activations.elu,
                kernel_initializer=initializers.VarianceScaling(
                    scale=1.0, mode="fan_in", distribution="truncated_normal"
                ),
                use_bias=True,
            ),
            layers.Dense(
                256,
                activation=activations.elu,
                kernel_initializer=initializers.VarianceScaling(
                    scale=1.0, mode="fan_in", distribution="truncated_normal"
                ),
                use_bias=True,
            ),
            layers.Dense(
                64,
                activation=activations.elu,
                kernel_initializer=initializers.VarianceScaling(
                    scale=1.0, mode="fan_in", distribution="truncated_normal"
                ),
                use_bias=True,
            ),
            layers.Dense(
                num_actions,
                activation=activations.linear,
                kernel_initializer=initializers.RandomUniform(
                    minval=-0.05, maxval=0.05
                ),
                use_bias=True,
                bias_initializer=initializers.Constant(-0.2),
            ),
        ]
        return layer_list

    @staticmethod
    def get_optimizer():
        return optimizers.Adam(learning_rate=0.0025)

    def create_agent(self, q_net, optimizer, train_step_counter):
        return DqnAgent(
            self.environment.time_step_spec(),
            self.environment.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            train_step_counter=train_step_counter,
            td_errors_loss_fn=losses.MeanSquaredError(),
            gamma=0.5,
        )

    def get_replay_buffer(self) -> ReplayBuffer:
        buffer_max_capacity = 20_000

        return TFUniformReplayBuffer(
            self.agent.collect_data_spec,
            batch_size=self.environment.batch_size,
            max_length=buffer_max_capacity,
        )

    def get_replay_buffer_iterator(self) -> Iterator:
        batch_size = 64

        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
        ).prefetch(3)

        return iter(dataset)

    def get_random_driver(self) -> PyDriver:
        random_policy = RandomTFPolicy(
            self.environment.time_step_spec(), self.environment.action_spec()
        )
        return PyDriver(
            self.environment,
            PyTFEagerPolicy(
                random_policy, use_tf_function=True, batch_time_steps=False
            ),
            [self.replay_buffer.add_batch],
            max_steps=100,
        )

    def get_collect_driver(self) -> PyDriver:
        collect_steps_per_iteration = 1

        return PyDriver(
            self.environment,
            PyTFEagerPolicy(
                self.agent.collect_policy, use_tf_function=True, batch_time_steps=False
            ),
            [self.replay_buffer.add_batch],
            max_steps=collect_steps_per_iteration,
        )

    def fainted_value(self) -> float:
        return 2.0

    def hp_value(self) -> float:
        return 1.0

    def number_of_pokemons(self) -> int:
        return 6

    def starting_value(self) -> float:
        return 0.0

    def status_value(self) -> float:
        return 0.0

    def victory_value(self) -> float:
        return 30.0


class AlphaPokeDoubleDQN(AlphaPokeSingleDQN):
    def create_agent(self, q_net, optimizer, train_step_counter):
        return DdqnAgent(
            self.environment.time_step_spec(),
            self.environment.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            train_step_counter=train_step_counter,
            td_errors_loss_fn=losses.MeanSquaredError(),
            gamma=0.5,
        )
