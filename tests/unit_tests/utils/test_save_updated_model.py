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
import os
import pickle
import pytest

from io import BytesIO
from unittest.mock import patch

from utils.create_agent import create_agent
from utils.save_updated_model import update_model

_TEST_MODEL = {"test": 42}


def simulate_pickled_model():
    return BytesIO(pickle.dumps(_TEST_MODEL))


def get_mock_args():
    data = {
        "battle_format": "gen8randombattle",
        "start_listening": False,
        "max_concurrent_battles": 45,
    }
    return data


def test_update_model_failure():
    agent = create_agent("dad", **get_mock_args())
    with pytest.raises(RuntimeError):
        update_model(agent, "./models")  # noqa


def test_update_model_success():
    cli_names = [
        "simpleRL-best",
        "expertRL-best",
        "simpleSarsaStark-best",
        "expertSarsaStark-best",
    ]
    folders = ["simpleRL", "expertRL", "SarsaStark", "expertSarsaStark"]
    for name, folder in zip(cli_names, folders):
        with patch("builtins.open") as mock_open:
            mock_open.return_value = simulate_pickled_model()
            agent = create_agent(name, **get_mock_args())[0]
        with patch("builtins.open") as mock_open, patch("pickle.dump") as mock_dump:
            with mock_open() as file:
                mock_file = file
            update_model(agent, "./models")  # noqa
            args, _ = mock_dump.call_args
            assert args[0] == _TEST_MODEL
            assert args[1] == mock_file
            args, _ = mock_open.call_args
            assert args[1] == "wb"
            assert os.path.dirname(args[0]) == os.path.join(
                "./models", folder, "gen8randombattle"
            )
