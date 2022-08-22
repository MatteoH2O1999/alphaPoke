import json
import time
import requests

from poke_env.utils import to_id_str


def get_ratings(username, battle_format):
    done = False
    json_data = None
    while not done:
        try:
            json_data = requests.get(
                f"https://pokemonshowdown.com/users/{to_id_str(username)}.json"
            )
            done = True
        except requests.exceptions.SSLError:
            print("SSL Error...")
            time.sleep(10)
            print("Retrying...")
    assert json_data is not None
    data = json.loads(json_data.content)
    rating_data = data["ratings"][battle_format]
    for key, value in rating_data.items():
        if key == "gxe":
            rating_data[key] = float(value)
        else:
            rating_data[key] = round(float(value))
    return rating_data
