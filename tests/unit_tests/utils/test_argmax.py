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
import pytest

from utils import argmax


def test_argmax():
    test1 = [0, 35, 678, 5]
    assert argmax(test1) == 2
    test2 = []
    with pytest.raises(ValueError):
        argmax(test2)
    test3 = [56.7]
    assert argmax(test3) == 0
    test4 = [0, 0, 0, 0, 0, 0, 0, 0, 2.3, 1, 2.29]
    assert argmax(test4) == 8
