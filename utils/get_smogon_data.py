import json
import requests


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
            .replace("unova", "")
            .replace("pikachualola", "pikachu")
            .replace("partner", "")
            .replace("%", "")
        )
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
        try:
            value["items"]
        except KeyError:
            value["items"] = []
    return to_return
