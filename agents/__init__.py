#
# A pokémon showdown battle-bot project based on reinforcement learning techniques.
# Copyright (C) 2022 Matteo Dell'Acqua
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Package level variables

LEARNING_RATE_WHILE_PLAYING = 0.001
MIN_LEARNING_RATE_WHILE_TRAINING = 0.01
EPSILON_WHILE_TRAINING_AND_PLAYING = 0.005
MIN_EPSILON_WHILE_TRAINING = 0.01
SARSA_DISCOUNT_FACTOR = 0.2
MON_HP_REWARD = 0.1
MON_FAINTED_REWARD = 10
VICTORY_REWARD = 30
