import json
import time
import requests

from poke_env.utils import to_id_str


def get_ratings(username, battle_format):
    done = False
    data = None
    while not done:
        try:
            print(f"Getting data for account {username}...")
            json_data = requests.get(
                f"https://pokemonshowdown.com/users/{to_id_str(username)}.json"
            )
            assert json_data is not None
            print(f"Converting smogon json data to {username} data...")
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
