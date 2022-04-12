# Module containing production-level agents with neural networks
import numpy as np
import tensorflow as tf

from abc import ABC
from gym.spaces import Space, Dict, Box
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.player.baselines import (
    RandomPlayer,
    MaxBasePowerPlayer,
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
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.specs import tensor_spec
from typing import Iterator, Union, List

from agents.base_classes.dqn_player import DQNPlayer
from agents.seba import Seba

rewards = {
    "fainted_value": 0.0,
    "hp": 0.0,
    "number_of_pokemons": 6,
    "starting_value": 0.0,
    "victory_reward": 1.0,
}


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
        boolean_flags = np.full(5, False, dtype=bool)
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
        boolean_flags = Box(low=False, high=True, shape=(5,), dtype=bool)
        return Dict(
            {
                "dynamax_turns": dynamax_turns,
                "boolean_flags": boolean_flags,
            }
        )


class _ActivePokemonEmbedding:
    @staticmethod
    def embed_pokemon(battle: AbstractBattle):
        pass

    @staticmethod
    def get_embedding() -> Space:
        pass


class _PokemonEmbedding:
    @staticmethod
    def embed_pokemon(mon: Pokemon):
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


class AlphaPokeSingleEmbedded(DQNPlayer, ABC):
    def __init__(self, log_interval=1000, eval_interval=10_000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.format_is_doubles:
            raise NotImplementedError("Double battles are not supported by this class")
        self.log_int = log_interval
        self.eval_int = eval_interval

    def calc_reward(
        self, last_battle: AbstractBattle, current_battle: AbstractBattle
    ) -> float:
        return self.reward_computing_helper(
            current_battle,
            fainted_value=rewards["fainted"],
            hp_value=rewards["hp"],
            number_of_pokemons=rewards["number_of_pokemons"],
            starting_value=rewards["starting_value"],
            status_value=rewards["status_value"],
            victory_value=rewards["victory_reward"],
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
            "active_mon": _ActivePokemonEmbedding.embed_pokemon(battle),
            "player_mon_1": _PokemonEmbedding.embed_pokemon(non_active_player_mons[0]),
            "player_mon_2": _PokemonEmbedding.embed_pokemon(non_active_player_mons[1]),
            "player_mon_3": _PokemonEmbedding.embed_pokemon(non_active_player_mons[2]),
            "player_mon_4": _PokemonEmbedding.embed_pokemon(non_active_player_mons[3]),
            "player_mon_5": _PokemonEmbedding.embed_pokemon(non_active_player_mons[4]),
            "opponent_active_mon": _PokemonEmbedding.embed_pokemon(
                battle.opponent_active_pokemon
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
                "opponent_active_mon": _PokemonEmbedding.get_embedding(),
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
                1024,
                activation=activations.elu,
                kernel_initializer=initializers.VarianceScaling(
                    scale=1.0, mode="fan_in", distribution="truncated_normal"
                ),
            ),
            layers.Dense(
                512,
                activation=activations.elu,
                kernel_initializer=initializers.VarianceScaling(
                    scale=1.0, mode="fan_in", distribution="truncated_normal"
                ),
            ),
            layers.Dense(
                256,
                activation=activations.elu,
                kernel_initializer=initializers.VarianceScaling(
                    scale=1.0, mode="fan_in", distribution="truncated_normal"
                ),
            ),
            layers.Dense(
                128,
                activation=activations.elu,
                kernel_initializer=initializers.VarianceScaling(
                    scale=1.0, mode="fan_in", distribution="truncated_normal"
                ),
            ),
            layers.Dense(
                64,
                activation=activations.elu,
                kernel_initializer=initializers.VarianceScaling(
                    scale=1.0, mode="fan_in", distribution="truncated_normal"
                ),
            ),
            layers.Dense(
                32,
                activation=activations.elu,
                kernel_initializer=initializers.VarianceScaling(
                    scale=1.0, mode="fan_in", distribution="truncated_normal"
                ),
            ),
            layers.Dense(
                num_actions,
                activation=activations.linear,
                kernel_initializer=initializers.RandomUniform(
                    minval=-0.05, maxval=0.05
                ),
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
