import json
import json5
import requests

from enum import Enum
from functools import lru_cache
from poke_env.data import GEN_TO_POKEDEX
from poke_env.utils import to_id_str

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
    pokedex = GEN_TO_POKEDEX[gen]
    abilities = []
    for pokemon in pokedex.values():
        for ability in pokemon["abilities"].values():
            if len(ability) > 0 and to_id_str(ability) not in abilities:
                abilities.append(to_id_str(ability))
    abilities.sort()

    return Enum("Abilities", abilities)  # noqa: functional API


@lru_cache(1)
def get_items():
    data = requests.get("https://play.pokemonshowdown.com/data/text/items.json5")
    data = json5.loads(data.content)
    json_items = list(data.keys())
    items = []
    for item in json_items:
        items.append(to_id_str(item))
    items.sort()
    return Enum("Items", items)  # noqa: functional API
