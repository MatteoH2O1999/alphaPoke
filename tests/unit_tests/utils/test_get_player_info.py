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
from unittest.mock import patch

from utils.get_player_info import get_ratings


class DummyContent:
    def __init__(self, content):
        self.content = content


def dummy_json():
    return DummyContent(
        b'{"username":"Training Account 1","userid":"trainingaccount1","registertime":1643760000,"group":1,'
        b'"ratings":{"gen8randombattle":{"elo":"1059.1815048809","gxe":"16.5","rpr":"1193.9405837661",'
        b'"rprd":"33.445757649977"},"ou":{"elo":"1157.9247205377","gxe":"65.9","rpr":"1629.4398628732",'
        b'"rprd":"101.38521808478"}}}'
    )


def test_get_ratings():
    actual_usernames = ["Test Account", "TestAccount", "testaccount"]
    expected_usernames = ["testaccount", "testaccount", "testaccount"]
    for actual, expected in zip(actual_usernames, expected_usernames):
        with patch("requests.get") as mock_request_get:
            mock_request_get.return_value = dummy_json()
            data = get_ratings(actual, "gen8randombattle")
            mock_request_get.assert_called_once_with(
                f"https://pokemonshowdown.com/users/{expected}.json"
            )
            assert data["elo"] == 1059
            assert data["gxe"] == 16.5
            assert data["rpr"] == 1194
            assert data["rprd"] == 33
