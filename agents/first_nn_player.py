# First neural network player following poke env tutorial
import numpy as np
import tensorflow as tf

from gym.spaces import Space, Box
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.baselines import RandomPlayer
from poke_env.player.openai_api import ObservationType
from poke_env.player.player import Player
from tensorflow.keras import activations, initializers, layers, optimizers
from tf_agents.agents import TFAgent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.drivers.py_driver import PyDriver
from tf_agents.networks import sequential
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from typing import Iterator, Union, List

from agents.base_classes.dqn_player import DQNPlayer


class FirstNNPlayer(DQNPlayer):
    def calc_reward(
        self, last_battle: AbstractBattle, current_battle: AbstractBattle
    ) -> float:
        victory_reward = 30.0
        mon_fainted_reward = 2.0
        mon_hp_reward = 0.1

        reward = 0
        if current_battle.won:
            reward += victory_reward
        elif current_battle.lost:
            reward -= victory_reward
        player_mons_ids = list(current_battle.team.keys())
        opponent_mons_ids = list(current_battle.opponent_team.keys())
        for mon_id in player_mons_ids:
            if (
                not last_battle.team[mon_id].fainted
                and current_battle.team[mon_id].fainted
            ):
                reward -= mon_fainted_reward
                reward -= (
                    mon_hp_reward * last_battle.team[mon_id].current_hp_fraction * 100
                )
            elif (
                not last_battle.team[mon_id].fainted
                and not current_battle.team[mon_id].fainted
            ):
                diff = (
                    last_battle.team[mon_id].current_hp_fraction
                    - current_battle.team[mon_id].current_hp_fraction
                )
                reward -= mon_hp_reward * diff * 100
        for mon_id in opponent_mons_ids:
            if mon_id not in last_battle.opponent_team.keys():
                if current_battle.opponent_team[mon_id].fainted:
                    reward += mon_hp_reward * 100 + mon_fainted_reward
                else:
                    reward += (
                        mon_hp_reward
                        * 100
                        * (1 - current_battle.opponent_team[mon_id].current_hp_fraction)
                    )
            else:
                if (
                    current_battle.opponent_team[mon_id].fainted
                    and not last_battle.opponent_team[mon_id].fainted
                ):
                    reward += mon_fainted_reward + (
                        100
                        * mon_hp_reward
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
                    reward += mon_hp_reward * 100 * diff
        return reward

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    @property
    def embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def opponents(self) -> Union[Player, str, List[Player], List[str]]:
        return RandomPlayer(battle_format=self.battle_format)

    def get_agent(self) -> TFAgent:
        action_tensor_spec = tensor_spec.from_spec(self.environment.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        network_layers = [
            layers.Dense(
                128,
                activation=activations.elu,
                kernel_initializer=initializers.VarianceScaling(
                    scale=2.0, mode="fan_in", distribution="truncated_normal"
                ),
            ),
            layers.Dense(
                128,
                activation=activations.elu,
                kernel_initializer=initializers.VarianceScaling(
                    scale=2.0, mode="fan_in", distribution="truncated_normal"
                ),
            ),
            layers.Dense(
                64,
                activation=activations.elu,
                kernel_initializer=initializers.VarianceScaling(
                    scale=2.0, mode="fan_in", distribution="truncated_normal"
                ),
            ),
            layers.Dense(
                num_actions,
                activation=activations.linear,
                kernel_initializer=initializers.VarianceScaling(
                    scale=2.0, mode="fan_in", distribution="truncated_normal"
                ),
                bias_initializer=initializers.Constant(-0.2),
            ),
        ]

        q_net = sequential.Sequential(network_layers)
        optimizer = optimizers.Adam(learning_rate=0.00025)

        train_step_counter = tf.Variable(0)

        agent = dqn_agent.DdqnAgent(
            self.environment.time_step_spec(),
            self.environment.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter,
            gamma=0.5,
        )
        agent.initialize()

        return agent

    def get_replay_buffer(self) -> ReplayBuffer:
        replay_buffer_capacity = 10_000

        return TFUniformReplayBuffer(
            self.agent.collect_data_spec,
            batch_size=self.environment.batch_size,
            max_length=replay_buffer_capacity,
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

    @property
    def log_interval(self) -> int:
        return 1000

    @property
    def eval_interval(self) -> int:
        return 10_000

    def eval_function(self, step):
        num_episodes = 100

        total_return = 0.0
        policy = self.agent.policy
        eval_env, _ = self.create_evaluation_env()
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            episode_return = 0.0
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        if "returns" not in self.evaluations.keys():
            self.evaluations["returns"] = []
        self.evaluations["returns"].append(avg_return.numpy()[0])
        eval_env.close()
        print(f"step = {step}: average return = {avg_return.numpy()[0]}")

    def log_function(self, step, loss_info: LossInfo):
        print(f"step = {step}: loss = {loss_info.loss}")
