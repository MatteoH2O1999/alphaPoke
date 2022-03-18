import json
import requests

from poke_env.utils import to_id_str


def get_ratings(username, battle_format):
    json_data = requests.get(
        f"https://pokemonshowdown.com/users/{to_id_str(username)}.json"
    )
    data = json.loads(json_data.content)
    rating_data = data["ratings"][battle_format]
    for key, value in rating_data.items():
        if key == "gxe":
            rating_data[key] = float(value)
        else:
            rating_data[key] = round(float(value))
    return rating_data
