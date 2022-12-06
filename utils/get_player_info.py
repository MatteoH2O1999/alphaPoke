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
import json
import time
import requests

from poke_env.utils import to_id_str


def get_ratings(username, battle_format):
    done = False
    data = None
    while not done:
        try:
            json_data = requests.get(
                f"https://pokemonshowdown.com/users/{to_id_str(username)}.json"
            )
            assert json_data is not None
            data = json.loads(json_data.content)
            assert data is not None
            done = True
        except requests.exceptions.SSLError:
            print("SSL Error...")
            time.sleep(10)
            print("Retrying...")
        except requests.exceptions.ConnectionError:
            print("Connection Error...")
            time.sleep(10)
            print("Retrying...")
        except json.JSONDecodeError:
            print("JSONDecodeError...")
            time.sleep(10)
            print("Retrying...")
    assert data is not None
    rating_data = data["ratings"][battle_format]
    for key, value in rating_data.items():
        if key == "gxe":
            rating_data[key] = float(value)
        else:
            rating_data[key] = round(float(value))
    return rating_data
