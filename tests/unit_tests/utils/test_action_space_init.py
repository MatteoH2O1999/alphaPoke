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
from utils.action_space_init import init_action_space


def test_init_actin_space():
    assert init_action_space(0) == []
    for i in range(1, 50):
        assert init_action_space(i) == [0 for i in range(i)]
