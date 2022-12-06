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
from poke_env.data import GEN_TO_POKEDEX
from unittest.mock import MagicMock, patch

import utils.get_smogon_data

from utils.get_smogon_data import (
    MEGA_STONES,
    Z_CRYSTALS,
    get_abilities,
    get_items,
    get_random_battle_learnset,
)


def test_get_random_battle_learnset_gen8():
    with patch("requests.get") as mock_req_get, patch("json.loads") as mock_loads:
        data = MagicMock()
        mock_req_get.return_value = data
        dummy_dict = {"pikachu": {}, "Mimikyu": {}, "Rattata": {}}
        mock_loads.return_value = dummy_dict
        learnset = get_random_battle_learnset(8)
        mock_req_get.assert_called_once_with(
            "https://raw.githubusercontent.com/pkmn/randbats/main/data/gen8randombattle.json"
        )
        mock_loads.assert_called_once_with(data.content)
        assert "pikachukalos" in learnset.keys()
        assert "mimikyubusted" in learnset.keys()
        assert "rattata" in learnset.keys()
        assert "eiscuenoice" not in learnset.keys()


def test_get_random_battle_learnset_gen6():
    with patch("requests.get") as mock_req_get, patch("json.loads") as mock_loads:
        data = MagicMock()
        mock_req_get.return_value = data
        dummy_dict = {"Gastrodon": {}, "Zygarde": {"items": ["item"]}, "Eiscue": {}}
        mock_loads.return_value = dummy_dict
        learnset = get_random_battle_learnset(6)
        mock_req_get.assert_called_once_with(
            "https://raw.githubusercontent.com/pkmn/randbats/main/data/gen6randombattle.json"
        )
        mock_loads.assert_called_once_with(data.content)
        assert "gastrodoneast" in learnset.keys()
        assert "eiscuenoice" in learnset.keys()
        assert "zygardecomplete" in learnset.keys()
        assert "pikachupartner" not in learnset.keys()
        assert learnset["gastrodoneast"]["items"] == []
        assert learnset["zygarde"]["items"] == ["item"]
        assert learnset["zygardecomplete"]["items"] == ["item"]


def get_fake_dict():
    return {
        6: {
            "Pokemon 1": {"abilities": {"H": "Ability 1", "1": "Ability 2"}},
            "Pokemon 2": {"abilities": {"H": "Ability 2", "1": "Ability 3"}},
        },
        8: {
            "Pokemon 1": {"abilities": {"H": "Ability 2", "1": "Ability 3"}},
            "Pokemon 2": {
                "abilities": {"H": "Ability 3", "1": "Ability 4", "2": "Ability 5"}
            },
        },
    }


def test_get_abilities_gen6():
    vars(utils.get_smogon_data).pop("Abilities6")
    tmp = GEN_TO_POKEDEX.copy()
    GEN_TO_POKEDEX.clear()
    GEN_TO_POKEDEX.update(get_fake_dict())
    abilities = list([v.name for v in get_abilities(6)])
    assert len(get_abilities(6)) == 3
    assert abilities == ["ability1", "ability2", "ability3"]
    GEN_TO_POKEDEX.clear()
    GEN_TO_POKEDEX.update(tmp)
    vars(utils.get_smogon_data).pop("Abilities6")
    get_abilities(6)


def test_get_abilities_gen8():
    vars(utils.get_smogon_data).pop("Abilities8")
    tmp = GEN_TO_POKEDEX.copy()
    GEN_TO_POKEDEX.clear()
    GEN_TO_POKEDEX.update(get_fake_dict())
    abilities = list([v.name for v in get_abilities(8)])
    assert len(get_abilities(8)) == 4
    assert abilities == ["ability2", "ability3", "ability4", "ability5"]
    GEN_TO_POKEDEX.clear()
    GEN_TO_POKEDEX.update(tmp)
    vars(utils.get_smogon_data).pop("Abilities8")
    get_abilities(8)


def get_fake_json():
    return {"item1": None, "item2": None, "Item3": None}


def test_get_items():
    vars(utils.get_smogon_data).pop("Items")
    with patch("requests.get") as mock_req_get, patch("json5.loads") as mock_loads:
        data = MagicMock()
        mock_req_get.return_value = data
        mock_loads.return_value = get_fake_json()
        items = get_items()
        assert len(items) == 3
        assert [i.name for i in items] == ["item1", "item2", "item3"]
        mock_req_get.assert_called_once()
        mock_loads.assert_called_once_with(data.content)
    vars(utils.get_smogon_data).pop("Items")
    get_items()


def test_mega_stones():
    for stone in MEGA_STONES:
        assert stone in get_items().__members__, f"{stone} not in possible items"


def test_z_crystals():
    for crystal in Z_CRYSTALS:
        assert crystal in get_items().__members__, f"{crystal} not in possible items"
