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
    "Dad": (
        "dad",
        DAD_DESCRIPTION,
        DAD_SUPPORTED_BATTLES,
    ),
    "8-Year-Old me": (
        "8-year-old-me",
        EIGHT_YEAR_OLD_ME_DESCRIPTION,
        EIGHT_YEAR_OLD_ME_SUPPORTED_BATTLES,
    ),
    "20-Year-Old me": (
        "20-year-old-me",
        TWENTY_YEAR_OLD_ME_DESCRIPTION,
        TWENTY_YEAR_OLD_ME_SUPPORTED_BATTLES,
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

LABELS = {
    "username": "Username:",
    "password": "Password:",
    "opponent_type": "Opponent type:",
    "battle_format": "Battle format:",
}
LABEL_WIDTH = max([len(label) for label in LABELS.values()])

OPTIONS = ["Select the opponent type", "Select a battle format"]
OPTIONS.extend(list(PLAYER_TYPE_DICT.keys()))
for _, _, battle_formats in PLAYER_TYPE_DICT.values():
    for f in battle_formats:
        if f not in OPTIONS:
            OPTIONS.append(f)
OPTION_WIDTH = max([len(label) for label in OPTIONS])


@lru_cache(maxsize=4)
def get_agent(cli_name, battle_format, username, password, start_timer):
    player_config = PlayerConfiguration(username, password)
    return create_agent(
        cli_name, battle_format, player_config, ShowdownServerConfiguration, start_timer
    )[0]


def setup_player_frame(frame: ttk.LabelFrame):
    username_label = ttk.Label(
        frame, text=LABELS["username"], anchor="e", width=LABEL_WIDTH
    )
    username_label.grid(row=0, column=0, padx=(10, 10), pady=(5, 5))
    username_input = ttk.Entry(frame, width=20)
    username_input.grid(row=0, column=1, padx=(10, 10), pady=(5, 5))
    password_label = ttk.Label(
        frame, text=LABELS["password"], anchor="e", width=LABEL_WIDTH
    )
    password_label.grid(row=1, column=0, padx=(10, 10), pady=(5, 5))
    password_input = ttk.Entry(frame, show="*", width=20)
    password_input.grid(row=1, column=1, padx=(10, 10), pady=(5, 5))


def setup_opponent_frame(frame: ttk.LabelFrame):
    def update_opponent_choice(new_value: tk.StringVar):
        opponent_description_text.config(state="normal")
        opponent_description_text.delete(1.0, tk.END)
        new_description = PLAYER_TYPE_DICT[new_value][1]
        formats = PLAYER_TYPE_DICT[new_value][2]
        opponent_description_text.insert(tk.END, new_description)
        height = 1
        opponent_description_text.config(height=1)
        opponent_description_text.update()
        while opponent_description_text.dlineinfo("1.end") is None:
            height += 1
            opponent_description_text.config(height=height)
            opponent_description_text.update()
        opponent_description_text.config(state="disabled")
        update_battle_formats(formats)

    def update_battle_formats(formats):
        if len(formats) == 0:
            battle_format.set(OPTIONS[1])
            battle_format_choice.config(state='disabled')
        else:
            previous_choice = battle_format.get()
            menu = battle_format_choice['menu']
            menu.delete(0, 'end')
            for f in formats:
                menu.add_command(label=f, command=lambda value=f: battle_format.set(value))
            if previous_choice in formats:
                battle_format.set(previous_choice)
            else:
                battle_format.set(OPTIONS[1])
            battle_format_choice.config(state='normal')

    opponent_label = ttk.Label(
        frame, text=LABELS["opponent_type"], anchor="e", width=LABEL_WIDTH
    )
    opponent_label.grid(row=0, column=0, padx=(10, 10), pady=(5, 5))
    possible_opponents = list(PLAYER_TYPE_DICT.keys())
    opponent = tk.StringVar()
    opponent_choice = ttk.OptionMenu(
        frame, opponent, OPTIONS[0], *possible_opponents, command=update_opponent_choice
    )
    opponent_choice.config(width=OPTION_WIDTH)
    opponent_choice.grid(row=0, column=1, padx=(10, 10), pady=(5, 5))
    battle_format_label = ttk.Label(
        frame, text=LABELS["battle_format"], anchor="e", width=LABEL_WIDTH
    )
    battle_format_label.grid(row=1, column=0, padx=(10, 10), pady=(5, 5))
    battle_format = tk.StringVar()
    battle_format_choice = ttk.OptionMenu(
        frame, battle_format, "Select a battle format"
    )
    battle_format_choice.grid(row=1, column=1, padx=(10, 10), pady=(5, 5))
    battle_format_choice.config(state="disabled", width=OPTION_WIDTH)
    opponent_description_text = tk.Text(frame, width=0, height=1, wrap=tk.WORD)
    opponent_description_text.config(state="disabled")
    opponent_description_text.grid(
        row=2, column=0, columnspan=2, padx=(10, 10), pady=(5, 5), sticky="we"
    )


def setup_buttons_frame(frame: ttk.LabelFrame):
    # TODO
    placeholder = ttk.Label(frame, text="button frame")
    placeholder.pack()


def setup_main_app(root):
    root.iconbitmap("./resources/icon.ico")
    root.title("alphaPoke AI")
    root.resizable(False, False)
    player_frame = ttk.LabelFrame(root, text="Bot account info")
    setup_player_frame(player_frame)
    opponent_frame = ttk.LabelFrame(root, text="Choose your opponent")
    setup_opponent_frame(opponent_frame)
    buttons_frame = ttk.LabelFrame(root)
    setup_buttons_frame(buttons_frame)
    player_frame.pack(fill="x", padx=(7, 7), pady=(10, 3))
    opponent_frame.pack(fill="x", padx=(7, 7), pady=(3, 3))
    buttons_frame.pack(fill="x", padx=(7, 7), pady=(3, 3))
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    app = tk.Tk()
    setup_main_app(app)
    app.mainloop()
