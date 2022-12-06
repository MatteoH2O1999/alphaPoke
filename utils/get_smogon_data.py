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
import json5
import requests

from enum import Enum
from functools import lru_cache
from poke_env.data import GEN_TO_POKEDEX
from poke_env.utils import to_id_str

GENERATIONS = 8

MEGA_STONES = [
    "abomasite",
    "absolite",
    "aerodactylite",
    "aggronite",
    "alakazite",
    "altarianite",
    "ampharosite",
    "audinite",
    "banettite",
    "beedrillite",
    "blastoisinite",
    "blazikenite",
    "cameruptite",
    "charizarditex",
    "charizarditey",
    "diancite",
    "galladite",
    "garchompite",
    "gardevoirite",
    "gengarite",
    "glalitite",
    "gyaradosite",
    "heracronite",
    "houndoominite",
    "kangaskhanite",
    "latiasite",
    "latiosite",
    "lopunnite",
    "lucarionite",
    "manectite",
    "mawilite",
    "medichamite",
    "metagrossite",
    "mewtwonitex",
    "mewtwonitey",
    "pidgeotite",
    "pinsirite",
    "sablenite",
    "salamencite",
    "sceptilite",
    "scizorite",
    "sharpedonite",
    "slowbronite",
    "steelixite",
    "swampertite",
    "tyranitarite",
    "venusaurite",
]

Z_CRYSTALS = [
    "aloraichiumz",
    "buginiumz",
    "darkiniumz",
    "decidiumz",
    "dragoniumz",
    "eeviumz",
    "electriumz",
    "fairiumz",
    "fightiniumz",
    "firiumz",
    "flyiniumz",
    "ghostiumz",
    "grassiumz",
    "groundiumz",
    "iciumz",
    "inciniumz",
    "kommoniumz",
    "lunaliumz",
    "lycaniumz",
    "marshadiumz",
    "mewniumz",
    "mimikiumz",
    "normaliumz",
    "pikaniumz",
    "pikashuniumz",
    "poisoniumz",
    "primariumz",
    "psychiumz",
    "rockiumz",
    "snorliumz",
    "solganiumz",
    "steeliumz",
    "tapuniumz",
    "ultranecroziumz",
    "wateriumz",
]


@lru_cache(8)
def get_random_battle_learnset(gen: int):
    data = requests.get(
        f"https://raw.githubusercontent.com/pkmn/randbats/main/data/gen{gen}randombattle.json"
    )
    data = json.loads(data.content)
    to_return = {}
    for key, value in data.items():
        new_key = (
            key.lower()
            .replace("-", "")
            .replace("school", "")
            .replace("gmax", "")
            .replace(" ", "")
            .replace(".", "")
            .replace(":", "")
            .replace("'", "")
            .replace("’", "")
            .replace("%", "")
        )
        try:
            value["items"]
        except KeyError:
            value["items"] = []
        if new_key == "mimikyu":
            to_return["mimikyubusted"] = value
        if new_key == "eiscue":
            to_return["eiscuenoice"] = value
        if new_key == "pikachu":
            to_return["pikachukalos"] = value
            to_return["pikachualola"] = value
            to_return["pikachugalar"] = value
            to_return["pikachuunova"] = value
            to_return["pikachupartner"] = value
            to_return["pikachuworld"] = value
            to_return["pikachujohto"] = value
            to_return["pikachuhoenn"] = value
            to_return["pikachusinnoh"] = value
            to_return["pikachuoriginal"] = value
            to_return["pikachukanto"] = value
        if new_key == "zygarde":
            to_return["zygardecomplete"] = value
        if new_key == "gastrodon":
            to_return["gastrodonwest"] = value
            to_return["gastrodoneast"] = value
        to_return[new_key] = value
    return to_return


def get_abilities(gen: int):
    if f"Abilities{gen}" in globals().keys():
        return globals()[f"Abilities{gen}"]
    pokedex = GEN_TO_POKEDEX[max(gen, 4)]
    abilities = []
    for pokemon in pokedex.values():
        for ability in pokemon["abilities"].values():
            if len(ability) > 0 and to_id_str(ability) not in abilities:
                abilities.append(to_id_str(ability))
    abilities.sort()
    to_return = Enum(f"Abilities{gen}", abilities, module=__name__)
    globals()[f"Abilities{gen}"] = to_return
    return to_return


def get_items():
    if "Items" in globals().keys():
        return globals()["Items"]
    data = requests.get("https://play.pokemonshowdown.com/data/text/items.json5")
    data = json5.loads(data.content)
    json_items = list(data.keys())
    items = []
    for item in json_items:
        items.append(to_id_str(item))
    items.sort()
    to_return = Enum("Items", items, module=__name__)
    globals()["Items"] = to_return
    return to_return


get_items()
for i in range(1, GENERATIONS, 1):
    get_abilities(i)
