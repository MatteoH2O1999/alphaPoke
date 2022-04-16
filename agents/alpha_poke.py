# Module containing production-level agents with neural networks
import numpy as np
import tensorflow as tf

from abc import ABC
from gym.spaces import Space, Dict, Box
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.effect import Effect
from poke_env.environment.field import Field
from poke_env.environment.move import Move
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
from poke_env.player.utils import background_evaluate_player
from tensorflow.keras import activations, initializers, layers, losses, optimizers
from tf_agents.agents import TFAgent
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.drivers.py_driver import PyDriver
from tf_agents.networks.sequential import Sequential
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
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

INFINITE_WEATHER = [Weather.DELTASTREAM, Weather.PRIMORDIALSEA, Weather.DESOLATELAND]

ABILITIES = get_abilities(8)

ITEMS = get_items()


class _BattlefieldEmbedding:
    @staticmethod
    def embed_battlefield(battle: AbstractBattle):
        battlefield_dict = {}
        dynamax_turns = np.full(2, -1, dtype=int)
        if battle.dynamax_turns_left is not None:
            dynamax_turns[0] = battle.dynamax_turns_left
        if battle.opponent_dynamax_turns_left is not None:
            dynamax_turns[1] = battle.opponent_dynamax_turns_left
        battlefield_dict["dynamax_turns"] = dynamax_turns
        boolean_flags = np.full(6, False, dtype=bool)
        if battle.can_mega_evolve:
            boolean_flags[0] = True
        if battle.can_z_move:
            boolean_flags[1] = True
        if battle.can_dynamax:
            boolean_flags[2] = True
        if battle.opponent_can_dynamax:
            boolean_flags[3] = True
        if battle.maybe_trapped:
            boolean_flags[4] = True
        if battle.force_switch:
            boolean_flags[5] = True
        battlefield_dict["booleans_flags"] = boolean_flags
        return battlefield_dict

    @staticmethod
    def get_embedding():
        dynamax_turns_low = [-1, -1]
        dynamax_turns_high = [3, 3]
        dynamax_turns = Box(
            low=np.array(dynamax_turns_low, dtype=int),
            high=np.array(dynamax_turns_high, dtype=int),
            dtype=int,
        )
        boolean_flags = Box(low=False, high=True, shape=(6,), dtype=bool)
        return Dict(
            {
                "dynamax_turns": dynamax_turns,
                "boolean_flags": boolean_flags,
            }
        )


class _ActivePokemonEmbedding:
    @staticmethod
    def embed_pokemon(mon: Pokemon, battle: AbstractBattle):
        pass

    @staticmethod
    def get_embedding() -> Space:
        pass


class _PokemonEmbedding:
    @staticmethod
    def embed_pokemon(mon: Pokemon, battle: AbstractBattle):
        pass

    @staticmethod
    def get_embedding() -> Space:
        pass


class _EnemyActivePokemonEmbedding:
    @staticmethod
    def embed_pokemon(mon: Pokemon, battle: AbstractBattle):
        pass

    @staticmethod
    def get_embedding() -> Space:
        pass


class _EnemyPokemonEmbedding:
    @staticmethod
    def embed_pokemon(mon: Pokemon):
        pass

    @staticmethod
    def get_embedding() -> Space:
        pass


class _MoveEmbedding:
    @staticmethod
    def embed_move(move: Move):
        pass

    @staticmethod
    def get_embedding() -> Space:
        pass


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
            status = np.full(len(Status), -1)
            chance = np.full(len(Status), -1, dtype=np.float64)
        else:
            status = np.full(len(Status), 0)
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
            boosts = np.full(7, -7)
            chance = np.full(7, -1, dtype=np.float64)
        else:
            boosts = np.full(7, 0)
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
            self_boosts = np.full(7, -7)
            chance = np.full(7, -1, dtype=np.float64)
        else:
            self_boosts = np.full(7, 0)
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
            return np.full(len(PokemonType), -1)
        types = np.full(len(PokemonType), 0)
        if isinstance(mon_or_move, Move):
            battle_types = [mon_or_move.type]
        elif isinstance(mon_or_move, Pokemon):
            battle_types = mon_or_move.types
        else:
            raise RuntimeError(f"Expected Move or Pokemon, got {type(mon_or_move)}.")
        for mon_type in battle_types:
            if mon_type is not None:
                types[mon_type.value] = 1
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
            return np.full(len(ITEMS), -1)
        battle_item = mon.item
        items = np.full(len(ITEMS), 0)
        items[getattr(ITEMS, battle_item).value] = 1
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
            return np.full(len(ABILITIES), -1)
        battle_abilities = np.full(len(ABILITIES), 0)
        if mon.ability is None:
            possible_abilities = mon.possible_abilities
            if len(possible_abilities) == 1:
                for ability in possible_abilities:
                    battle_abilities[getattr(ABILITIES, ability).value] = 2
            else:
                for ability in possible_abilities:
                    battle_abilities[getattr(ABILITIES, ability).value] = 1
            return battle_abilities
        battle_abilities[getattr(ABILITIES, mon.ability).value] = 2
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
        weathers = np.full(len(Weather), -1)
        for w in INFINITE_WEATHER:
            weathers[w.value] = 0
        for w, value in weather.items():
            if w in INFINITE_WEATHER:
                weathers[w.value] = 1
            else:
                weathers[w.value] = current_turn - value
        return weathers

    @staticmethod
    def get_embedding() -> Space:
        low_bound = [-1 for _ in range(len(Weather))]
        high_bound = [6 for _ in range(len(Weather))]
        for w in INFINITE_WEATHER:
            high_bound[w.value] = 1
            low_bound[w.value] = 0
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
            statuses = np.full(len(Status), 0)
            if status is not None:
                statuses[status.value] = 1
        else:
            statuses = np.full(len(Status), -1)
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
        effects = np.full(len(Effect), -1)
        for effect, counter in battle_effects.items():
            effects[effect.value] = counter
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
        side_conditions = np.full(len(SideCondition), -1)
        side_conditions[SideCondition.STEALTH_ROCK.value] = 0
        for condition in STACKABLE_CONDITIONS.keys():
            side_conditions[condition.value] = 0
        for condition, value in battle_side_conditions.items():
            if condition in STACKABLE_CONDITIONS.keys():
                side_conditions[condition.value] = value
            elif condition == SideCondition.STEALTH_ROCK:
                side_conditions[condition.value] = 1
            else:
                side_conditions[condition.value] = current_turn - value
        return side_conditions

    @staticmethod
    def get_embedding() -> Space:
        low_bound = [-1 for _ in range(len(SideCondition))]
        high_bound = [6 for _ in range(len(SideCondition))]
        low_bound[SideCondition.STEALTH_ROCK.value] = 0
        high_bound[SideCondition.STEALTH_ROCK.value] = 1
        for condition in STACKABLE_CONDITIONS.keys():
            low_bound[condition.value] = 0
            high_bound[condition.value] = STACKABLE_CONDITIONS[condition]
        return Box(
            low=np.array(low_bound, dtype=int),
            high=np.array(high_bound, dtype=int),
            dtype=int,
        )


# One hot encoding for the fields.
class _FieldEmbedding:
    @staticmethod
    def embed_field(battle: AbstractBattle):
        current_turn = battle.turn
        fields = np.full(len(Field), -1)
        battle_fields = battle.fields
        for field, value in battle_fields.items():
            fields[field.value] = current_turn - value
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

    @property
    def fainted_value(self) -> float:
        return 0.0

    @property
    def hp_value(self) -> float:
        return 0.0

    @property
    def number_of_pokemons(self) -> int:
        return 6

    @property
    def starting_value(self) -> float:
        return 0.0

    @property
    def status_value(self) -> float:
        return 0.0

    @property
    def victory_value(self) -> float:
        return 1.0

    def calc_reward(
        self, last_battle: AbstractBattle, current_battle: AbstractBattle
    ) -> float:
        return self.reward_computing_helper(
            current_battle,
            fainted_value=self.fainted_value,
            hp_value=self.hp_value,
            number_of_pokemons=self.number_of_pokemons,
            starting_value=self.starting_value,
            status_value=self.status_value,
            victory_value=self.victory_value,
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
                non_active_opponent_mons[0]
            ),
            "opponent_mon_2": _EnemyPokemonEmbedding.embed_pokemon(
                non_active_opponent_mons[1]
            ),
            "opponent_mon_3": _EnemyPokemonEmbedding.embed_pokemon(
                non_active_opponent_mons[2]
            ),
            "opponent_mon_4": _EnemyPokemonEmbedding.embed_pokemon(
                non_active_opponent_mons[3]
            ),
            "opponent_mon_5": _EnemyPokemonEmbedding.embed_pokemon(
                non_active_opponent_mons[4]
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
        num_challenges = 2000
        placement_challenges = 40

        eval_env, agent = self.create_evaluation_env(active=False)
        policy = self.agent.policy
        task = background_evaluate_player(
            agent, n_battles=num_challenges, n_placement_battles=placement_challenges
        )
        for _ in range(num_challenges):
            time_step = eval_env.reset()
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step)
        evaluation = task.result()

        if "evaluations" not in self.evaluations.keys():
            self.evaluations["evaluations"] = [[], []]
        self.evaluations["evaluations"][0].append(step)
        self.evaluations["evaluations"][1].append(evaluation)

    def log_function(self, step, loss_info: LossInfo):
        if "losses" not in self.evaluations.keys():
            self.evaluations["losses"] = [[], []]
        self.evaluations["losses"][0].append(step)
        self.evaluations["losses"][1].append(loss_info.loss)


class AlphaPokeSingleDQN(AlphaPokeSingleEmbedded):
    def get_agent(self) -> TFAgent:
        action_tensor_spec = tensor_spec.from_spec(self.environment.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        q_net = Sequential(self.get_network_layers(num_actions))
        optimizer = self.get_optimizer()

        train_step_counter = tf.Variable(0)

        agent = self.create_agent(q_net, optimizer, train_step_counter)
        agent.initialise()
        return agent

    @staticmethod
    def get_network_layers(num_actions):
        layer_list = [
            layers.Dense(
                16384,
                activation=activations.elu,
                kernel_initializer=initializers.VarianceScaling(
                    scale=1.0, mode="fan_in", distribution="truncated_normal"
                ),
                use_bias=True,
            ),
            layers.Dense(
                4096,
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
                512,
                activation=activations.elu,
                kernel_initializer=initializers.VarianceScaling(
                    scale=1.0, mode="fan_in", distribution="truncated_normal"
                ),
                use_bias=True,
            ),
            layers.Dense(
                128,
                activation=activations.elu,
                kernel_initializer=initializers.VarianceScaling(
                    scale=1.0, mode="fan_in", distribution="truncated_normal"
                ),
                use_bias=True,
            ),
            layers.Dense(
                32,
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
