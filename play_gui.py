import asyncio
import os
import tkinter as tk
import tkinter.ttk as ttk
import sys

from functools import lru_cache
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration
from threading import Thread
from tkinter import messagebox

from gui_data import PLAYER_TYPE_DICT
from utils.create_agent import create_agent

ICON_PATH = "./resources/icon.ico"
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    ICON_PATH = os.path.join(sys._MEIPASS, "resources", "icon.ico")

LABELS = {
    "username": "Username:",
    "password": "Password:",
    "opponent_type": "Opponent type:",
    "battle_format": "Battle format:",
    "player_username": "Username to challenge:",
}
LABEL_WIDTH = max([len(label) for label in LABELS.values()])
ENTRY_WIDTH = 20

OPTIONS = ["Select the opponent type", "Select a battle format"]
OPTIONS.extend(list(PLAYER_TYPE_DICT.keys()))
for _, _, battle_formats in PLAYER_TYPE_DICT.values():
    for b_format in battle_formats:
        if b_format not in OPTIONS:
            OPTIONS.append(b_format)
OPTION_WIDTH = max([len(label) for label in OPTIONS])

BUTTON_LABELS = ["Send challenge to ", "Accept challenge from "]
BUTTON_WIDTH = max([len(label) for label in BUTTON_LABELS]) + 20

INTERACTIVE_WIDGETS = []
THREADS = []


def disable_everything():
    for w in INTERACTIVE_WIDGETS:
        w.configure(state=tk.DISABLED)


def restore_everything():
    for w in INTERACTIVE_WIDGETS:
        w.configure(state=tk.NORMAL)


class FormData:
    username: str = ""
    password: str = ""
    agent_type: str = ""
    battle_format: str = ""
    user_to_challenge: str = ""
    timer: bool = False


def update_username(*args):
    FORM_DATA.username = username.get()
    update_button_status()


def update_password(*args):
    FORM_DATA.password = password.get()
    update_button_status()


def update_opponent_variable(*args):
    FORM_DATA.agent_type = opponent.get()
    update_button_status()


def update_battle_format_variable(*args):
    FORM_DATA.battle_format = battle_format.get()
    update_button_status()


def update_timer(*args):
    FORM_DATA.timer = timer.get()
    update_button_status()


def update_username_to_challenge(*args):
    if len(username_to_challenge.get()) >= 20:
        username.set(username_to_challenge.get()[:20])
    if len(username_to_challenge.get()) > 0:
        send_challenges_text.set(f"Send challenge to {username_to_challenge.get()}")
        accept_challenges_text.set(
            f"Accept challenge from {username_to_challenge.get()}"
        )
    else:
        send_challenges_text.set("Send challenge")
        accept_challenges_text.set("Accept challenge")
    FORM_DATA.user_to_challenge = username_to_challenge.get()
    update_button_status()


def update_button_status():
    if (
        len(FORM_DATA.username) > 0
        and len(FORM_DATA.password) > 0
        and FORM_DATA.agent_type in PLAYER_TYPE_DICT.keys()
        and FORM_DATA.battle_format in PLAYER_TYPE_DICT[FORM_DATA.agent_type][2]
        and len(FORM_DATA.user_to_challenge) > 0
        and FORM_DATA.timer is not None
    ):
        send_button.config(state="normal")
        accept_button.config(state="normal")
    else:
        send_button.config(state="disabled")
        accept_button.config(state="disabled")


app = tk.Tk()
FORM_DATA = FormData()
username = tk.StringVar()
password = tk.StringVar()
opponent = tk.StringVar()
battle_format = tk.StringVar()
username_to_challenge = tk.StringVar()
timer = tk.BooleanVar()
send_challenges_text = tk.StringVar()
accept_challenges_text = tk.StringVar()
send_button: ttk.Button
accept_button: ttk.Button


@lru_cache(maxsize=4)
def get_agent(cli_name, battle_format, username, password, start_timer):
    player_config = PlayerConfiguration(username, password)
    return create_agent(
        cli_name, battle_format, player_config, ShowdownServerConfiguration, start_timer
    )[0]


def setup_player_frame(frame: ttk.LabelFrame):

    # Username input
    username_label = ttk.Label(
        frame, text=LABELS["username"], anchor="e", width=LABEL_WIDTH
    )
    username_label.grid(row=0, column=0, padx=(10, 10), pady=(5, 5))
    username_input = ttk.Entry(frame, width=ENTRY_WIDTH, textvariable=username)
    username_input.grid(row=0, column=1, padx=(10, 10), pady=(5, 5))

    # Password input
    password_label = ttk.Label(
        frame, text=LABELS["password"], anchor="e", width=LABEL_WIDTH
    )
    password_label.grid(row=1, column=0, padx=(10, 10), pady=(5, 5))
    password_input = ttk.Entry(
        frame, show="*", width=ENTRY_WIDTH, textvariable=password
    )
    password_input.grid(row=1, column=1, padx=(10, 10), pady=(5, 5))

    # Append interactive widgets
    INTERACTIVE_WIDGETS.append(username_input)
    INTERACTIVE_WIDGETS.append(password_input)


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
            battle_format_choice.config(state="disabled")
        else:
            previous_choice = battle_format.get()
            menu = battle_format_choice["menu"]
            menu.delete(0, "end")
            for f in formats:
                menu.add_command(
                    label=f, command=lambda value=f: battle_format.set(value)
                )
            if previous_choice in formats:
                battle_format.set(previous_choice)
            else:
                battle_format.set(OPTIONS[1])
            battle_format_choice.config(state="normal")

    # Opponent choice
    opponent_label = ttk.Label(
        frame, text=LABELS["opponent_type"], anchor="e", width=LABEL_WIDTH
    )
    opponent_label.grid(row=0, column=0, padx=(10, 10), pady=(5, 5))
    possible_opponents = list(PLAYER_TYPE_DICT.keys())
    opponent_choice = ttk.OptionMenu(
        frame, opponent, OPTIONS[0], *possible_opponents, command=update_opponent_choice
    )
    opponent_choice.config(width=OPTION_WIDTH)
    opponent_choice.grid(row=0, column=1, padx=(10, 10), pady=(5, 5))

    # Battle format choice
    battle_format_label = ttk.Label(
        frame, text=LABELS["battle_format"], anchor="e", width=LABEL_WIDTH
    )
    battle_format_label.grid(row=1, column=0, padx=(10, 10), pady=(5, 5))
    battle_format_choice = ttk.OptionMenu(
        frame, battle_format, "Select a battle format"
    )
    battle_format_choice.grid(row=1, column=1, padx=(10, 10), pady=(5, 5))
    battle_format_choice.config(state="disabled", width=OPTION_WIDTH)

    # Opponent description text
    opponent_description_text = tk.Text(
        frame, width=0, height=1, wrap=tk.WORD, padx=4, pady=2
    )
    opponent_description_text.config(state="disabled", font=("CourierNew", 10))
    opponent_description_text.grid(
        row=2, column=0, columnspan=2, padx=(10, 10), pady=(5, 5), sticky="we"
    )

    # Append interactive widgets
    INTERACTIVE_WIDGETS.append(opponent_choice)
    INTERACTIVE_WIDGETS.append(battle_format_choice)


def setup_buttons_frame(frame: ttk.LabelFrame):
    def send_challenge_click():
        t = Thread(target=send_challenge_thread, daemon=True)
        t.start()
        THREADS.append(t)

    def send_challenge_thread():
        asyncio.set_event_loop(asyncio.new_event_loop())
        agent_type_name = PLAYER_TYPE_DICT[FORM_DATA.agent_type][0]
        b_format = PLAYER_TYPE_DICT[FORM_DATA.agent_type][2][FORM_DATA.battle_format]
        agent = get_agent(
            agent_type_name,
            b_format,
            FORM_DATA.username,
            FORM_DATA.password,
            FORM_DATA.timer,
        )
        print(agent)
        disable_everything()
        asyncio.get_event_loop().run_until_complete(
            agent.send_challenges(FORM_DATA.user_to_challenge, 1)
        )
        restore_everything()

    def accept_challenge_click():
        t = Thread(target=accept_challenge_thread, daemon=True)
        t.start()
        THREADS.append(t)

    def accept_challenge_thread():
        asyncio.set_event_loop(asyncio.new_event_loop())
        agent_type_name = PLAYER_TYPE_DICT[FORM_DATA.agent_type][0]
        b_format = PLAYER_TYPE_DICT[FORM_DATA.agent_type][2][FORM_DATA.battle_format]
        agent = get_agent(
            agent_type_name,
            b_format,
            FORM_DATA.username,
            FORM_DATA.password,
            FORM_DATA.timer,
        )
        disable_everything()
        asyncio.get_event_loop().run_until_complete(
            agent.accept_challenges(FORM_DATA.user_to_challenge, 1)
        )
        restore_everything()

    # Helper frame
    helper_frame = ttk.Frame(frame)
    helper_frame.pack(fill="x")

    # Player username input
    player_username_label = ttk.Label(
        helper_frame, text=LABELS["player_username"], anchor="e", width=LABEL_WIDTH
    )
    player_username_label.grid(row=0, column=0, padx=(10, 10), pady=(5, 5))
    player_username_input = ttk.Entry(
        helper_frame, width=ENTRY_WIDTH, textvariable=username_to_challenge
    )
    player_username_input.grid(row=0, column=1, padx=(10, 10), pady=(5, 5))

    # Timer checkbox
    timer_checkbox = ttk.Checkbutton(
        helper_frame, text="Use battle timer", variable=timer
    )
    timer_checkbox.grid(row=0, column=2, padx=(10, 10), pady=(5, 5))
    timer.set(False)

    # Challenge buttons
    global send_button
    global accept_button
    send_challenges_text.set("Send challenge")
    send_challenges_button = ttk.Button(
        frame,
        padding=(3, 1, 3, 1),
        textvariable=send_challenges_text,
        width=BUTTON_WIDTH,
        command=send_challenge_click,
        state="disabled",
    )
    send_challenges_button.pack(padx=(10, 10), pady=(5, 5))
    accept_challenges_text.set("Accept challenge")
    accept_challenges_button = ttk.Button(
        frame,
        padding=(3, 1, 3, 1),
        textvariable=accept_challenges_text,
        width=BUTTON_WIDTH,
        command=accept_challenge_click,
        state="disabled",
    )
    accept_challenges_button.pack(padx=(10, 10), pady=(5, 5))
    send_button = send_challenges_button
    accept_button = accept_challenges_button

    # Append interactive widgets
    INTERACTIVE_WIDGETS.append(player_username_input)
    INTERACTIVE_WIDGETS.append(timer_checkbox)
    INTERACTIVE_WIDGETS.append(send_challenges_button)
    INTERACTIVE_WIDGETS.append(accept_challenges_button)


def setup_main_app(root):
    root.iconbitmap(ICON_PATH)
    root.title("alphaPoke AI")
    root.resizable(False, False)
    player_frame = ttk.LabelFrame(root, text="Bot account info")
    setup_player_frame(player_frame)
    opponent_frame = ttk.LabelFrame(root, text="Choose your opponent")
    setup_opponent_frame(opponent_frame)
    buttons_frame = ttk.LabelFrame(root, text="Challenge controls")
    setup_buttons_frame(buttons_frame)
    player_frame.pack(fill="x", padx=(7, 7), pady=(10, 3))
    opponent_frame.pack(fill="x", padx=(7, 7), pady=(3, 3))
    buttons_frame.pack(fill="x", padx=(7, 7), pady=(3, 3))
    username.trace_add("write", update_username)
    password.trace_add("write", update_password)
    opponent.trace_add("write", update_opponent_variable)
    battle_format.trace_add("write", update_battle_format_variable)
    username_to_challenge.trace_add("write", update_username_to_challenge)
    timer.trace_add("write", update_timer)


def on_closing():
    running = False
    for t in THREADS:
        if t.is_alive():
            running = True
    if running:
        if not messagebox.askokcancel(
            "Force quit",
            "There are still unfinished battles. "
            "Are you sure you want to shutdown all battles in the background?",
        ):
            return
    app.destroy()


if __name__ == "__main__":  # pragma: no cover
    setup_main_app(app)
    app.protocol("WM_DELETE_WINDOW", on_closing)
    app.mainloop()
