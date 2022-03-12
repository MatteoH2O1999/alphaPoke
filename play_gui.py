from functools import lru_cache
import tkinter as tk
import tkinter.ttk as ttk

from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration

from utils.create_agent import create_agent

DAD_DESCRIPTION = "dad description"
EIGHT_YEAR_OLD_ME_DESCRIPTION = "8 year old me description"
TWENTY_YEAR_OLD_ME_DESCRIPTION = "20 year old me description"
SIMPLE_RL_DESCRIPTION = "simple rl description"
EXPERT_RL_DESCRIPTION = "expert rl description"
SIMPLE_SARSA_DESCRIPTION = "simple sarsa description"
EXPERT_SARSA_DESCRIPTION = "expert sarsa description"

DAD_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}
EIGHT_YEAR_OLD_ME_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}
TWENTY_YEAR_OLD_ME_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}
SIMPLE_RL_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}
EXPERT_RL_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}
SIMPLE_SARSA_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}
EXPERT_SARSA_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}


PLAYER_TYPE_DICT = {
    "Dad": ("dad", DAD_DESCRIPTION),
    "8-Year-Old me": (
        "8-year-old-me",
        EIGHT_YEAR_OLD_ME_DESCRIPTION,
        DAD_SUPPORTED_BATTLES,
    ),
    "20-Year-Old me": (
        "20-year-old-me",
        TWENTY_YEAR_OLD_ME_DESCRIPTION,
        EIGHT_YEAR_OLD_ME_SUPPORTED_BATTLES,
    ),
    "Simple Reinforcement Learning Player": (
        "simpleRL-best",
        SIMPLE_RL_DESCRIPTION,
        SIMPLE_RL_SUPPORTED_BATTLES,
    ),
    "Expert Reinforcement Learning Player": (
        "expertRL-best",
        EXPERT_RL_DESCRIPTION,
        EXPERT_RL_SUPPORTED_BATTLES,
    ),
    "Sarsa Trained Player": (
        "simpleSarsaStark-best",
        SIMPLE_SARSA_DESCRIPTION,
        SIMPLE_SARSA_SUPPORTED_BATTLES,
    ),
    "Expert Sarsa Trained Player": (
        "expertSarsaStark-best",
        EXPERT_SARSA_DESCRIPTION,
        EXPERT_SARSA_SUPPORTED_BATTLES,
    ),
}


@lru_cache(maxsize=4)
def get_agent(cli_name, battle_format, username, password, start_timer):
    player_config = PlayerConfiguration(username, password)
    return create_agent(
        cli_name, battle_format, player_config, ShowdownServerConfiguration, start_timer
    )[0]


def setup_player_frame(frame: ttk.LabelFrame):
    username_label = ttk.Label(frame, text="Username", anchor="e")
    username_label.grid(row=0, column=0, padx=(10, 10), pady=(5, 5))
    username_input = ttk.Entry(frame, width=20)
    username_input.grid(row=0, column=1, padx=(10, 10), pady=(5, 5))
    password_label = ttk.Label(frame, text="Password", anchor="e")
    password_label.grid(row=1, column=0, padx=(10, 10), pady=(5, 5))
    password_input = ttk.Entry(frame, show="*", width=20)
    password_input.grid(row=1, column=1, padx=(10, 10), pady=(5, 5))


def setup_opponent_frame(frame: ttk.LabelFrame):
    # TODO
    placeholder = ttk.Label(frame, text="opponent frame")
    placeholder.pack()


def setup_buttons_frame(frame: ttk.LabelFrame):
    # TODO
    placeholder = ttk.Label(frame, text="button frame")
    placeholder.pack()


def main_app():
    root = tk.Tk()
    root.iconbitmap("./resources/icon.ico")
    root.title("alphaPoke AI")
    root.resizable(False, False)
    player_frame = ttk.LabelFrame(root, text="Bot account info")
    setup_player_frame(player_frame)
    opponent_frame = ttk.LabelFrame(root)
    setup_opponent_frame(opponent_frame)
    buttons_frame = ttk.LabelFrame(root)
    setup_buttons_frame(buttons_frame)
    player_frame.pack(fill="x", padx=(7, 7), pady=(10, 3))
    opponent_frame.pack(fill="x", padx=(7, 7), pady=(3, 3))
    buttons_frame.pack(fill="x", padx=(7, 7), pady=(3, 3))
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main_app()
