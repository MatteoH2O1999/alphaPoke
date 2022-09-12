# Allows bots to play on the ladder
import asyncio
import asyncio.exceptions
import datetime
import gc
import getpass
import math
import matplotlib.pyplot as plt
import multiprocessing
import os
import seaborn as sns
import sys
import time

from csv import DictReader
from numpy import mean
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import ForfeitBattleOrder, BattleOrder
from poke_env.server_configuration import ShowdownServerConfiguration
from poke_env.player.player import Player
from poke_env.player_configuration import PlayerConfiguration
from typing import Awaitable, Union

from agents.base_classes.trainable_player import TrainablePlayer
from utils.close_player import close_player
from utils.create_agent import create_agent
from utils.invalid_argument import InvalidArgumentNumber
from utils.get_player_info import get_ratings
from utils.save_updated_model import update_model


MAX_WAIT_TIME_FOR_ELO_UPDATE = 300
MAX_BATTLE_TIME = 3600


def main():
    if len(sys.argv) < 4:
        raise InvalidArgumentNumber()
    current_time = datetime.datetime.now()
    current_time_string = current_time.strftime("%d-%m-%Y %H-%M-%S")
    battle_format = sys.argv[1]
    save_replays = (
        sys.argv[2].lower() == "true"
        or sys.argv[2].lower() == "t"
        or sys.argv[2].lower() == "y"
    )
    save_path = sys.argv[3]
    number_of_challenges = None
    start_agent = 4
    if sys.argv[4].isdigit() and float(sys.argv[4]).is_integer():
        number_of_challenges = int(sys.argv[4])
        start_agent = 5
    processes = []
    max_completed_battles = multiprocessing.Value("i", 0)
    cont = multiprocessing.Value("i", 1)
    for i in range(start_agent, len(sys.argv)):
        processes.append(
            PlayerProcess(
                sys.argv[i],
                battle_format,
                save_replays,
                save_path,
                current_time_string,
                max_completed_battles,
                cont,
            )
        )
    for p in processes:
        p.start()
    if number_of_challenges:
        with max_completed_battles.get_lock():
            with cont.get_lock():
                max_completed_battles.value = number_of_challenges
                cont.value = 0
    else:
        input("Press any key to stop...")
        with cont.get_lock():
            cont.value = 0
    for p in processes:
        p.join()
    complete_data = []
    for file in os.listdir(save_path):
        if ".csv" in file:
            with open(os.path.join(save_path, file)) as agent_file:
                reader = DictReader(agent_file, delimiter=";")
                for row in reader:
                    complete_data.append(
                        (file.replace(".csv", ""), row["index_of_battle"], row["elo"])
                    )
            os.remove(os.path.join(save_path, file))
    with open(os.path.join(save_path, "data.csv"), "w") as data_file:
        data_file.write("agent;index_of_battle;elo\n")
        for data in complete_data:
            data_file.write(f"{data[0]};{data[1]};{data[2]}\n")


class PlayerProcess(multiprocessing.Process):
    def __init__(
        self,
        agent_type,
        battle_format,
        save_replays,
        save_path,
        plot_time,
        stop,
        cont,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cont = True
        self.cont_int = cont
        self.stop_on_shared = stop
        self.username = input("Account username: ")
        self.password = getpass.getpass()
        self.agent_type = agent_type
        self.battle_format = battle_format
        self.save_replays = save_replays
        self.save_path = save_path
        self.plot_time = plot_time
        self.agent = None
        self.plot_path = None
        self.count = 1
        self.stop_on = 1

    def run(self) -> None:
        print(f"Resetting elo for player {self.username}...")
        self.reset_elo()
        plt.switch_backend("agg")
        print(f"Creating agent of type {self.agent_type}...")
        self.agent = create_agent(
            self.agent_type,
            self.battle_format,
            PlayerConfiguration(self.username, self.password),
            ShowdownServerConfiguration,
            True,
            self.save_replays,
        )[0]
        file_name = f"rank {self.agent.__class__.__name__} {self.plot_time}.png"
        self.plot_path = os.path.join(self.save_path, file_name)
        elo_stats = [[0], [1000]]
        while self.cont or self.stop_on >= self.count:
            print(f"Starting battle {self.count} for agent {self.username}...")
            try:
                asyncio.get_event_loop().run_until_complete(
                    asyncio.wait_for(self.agent.ladder(1), MAX_BATTLE_TIME)
                )
            except asyncio.exceptions.TimeoutError:
                print(f"Error with battle {self.count} for agent {self.username}...")
                model = None
                if isinstance(self.agent, TrainablePlayer) and (
                    self.agent.training or self.agent.train_while_playing
                ):
                    model = self.agent.model
                print(f"Closing player {self.username}...")
                close_player(self.agent)
                print(
                    f"Creating new agent of type {self.agent_type} for account {self.username}"
                )
                self.agent = create_agent(
                    self.agent_type,
                    self.battle_format,
                    PlayerConfiguration(self.username, self.password),
                    ShowdownServerConfiguration,
                    True,
                    self.save_replays,
                )
                if model is not None:
                    assert isinstance(self.agent, TrainablePlayer)
                    self.agent.model = model
                gc.collect()
                continue
            print(f"Battle {self.count} finished for agent {self.username}...")
            elo_stats[0].append(self.count)
            last_elo = elo_stats[1][-1]
            print(f"Last elo for agent {self.username}: {last_elo}...")
            new_elo = get_ratings(self.username, self.battle_format)["elo"]
            counter = MAX_WAIT_TIME_FOR_ELO_UPDATE
            while new_elo == last_elo and counter > 0:
                print(f"Elo of agent {self.username} not updated yet, retrying...")
                time.sleep(20)
                new_elo = get_ratings(self.username, self.battle_format)["elo"]
                counter -= 20
            elo_stats[1].append(new_elo)
            print(f"Getting lock on stop_on and cont_int for player {self.username}...")
            with self.stop_on_shared.get_lock():
                with self.cont_int.get_lock():
                    if self.stop_on_shared.value < self.count:
                        print(
                            f"Updating stop_on value from {self.stop_on_shared.value} to {self.count}..."
                        )
                        self.stop_on_shared.value = self.count
                    if self.stop_on != self.stop_on_shared.value:
                        print(
                            f"Updating stop_on variable for agent {self.username} from {self.stop_on} to {self.stop_on_shared.value}"
                        )
                    self.stop_on = self.stop_on_shared.value
                    if self.cont_int.value == 1:
                        self.cont = True
                    elif self.cont_int.value == 0:
                        self.cont = False
                    else:
                        raise RuntimeError(
                            f"Invalid value. Expected 0 or 1, got {self.cont_int.value}"
                        )
                    print(
                        f"Value of cont variable for agent {self.username}: {self.cont}..."
                    )
            self.count += 1
        if isinstance(self.agent, TrainablePlayer) and (
            self.agent.training or self.agent.train_while_playing
        ):
            update_model(self.agent, "./models")
        print(f"Saving logs for player {self.username}...")
        with open(
            os.path.join(self.save_path, f"{self.agent.__class__.__name__}.csv"), "w"
        ) as data_file:
            data_file.write("index_of_battle;elo\n")
            for index_of_battle, elo in zip(elo_stats[0], elo_stats[1]):
                data_file.write(f"{index_of_battle};{elo}\n")
        os.makedirs(os.path.dirname(self.plot_path), exist_ok=True)
        means = []
        window_size = max(math.floor(math.log2(len(elo_stats[0]))), 1)
        if window_size % 2 == 0:
            window_size += 1
        window_half_size = (window_size - 1) // 2
        extended_stats = [
            *[elo_stats[1][0] for _ in range(window_half_size)],
            *elo_stats[1],
            *[elo_stats[1][-1] for _ in range(window_half_size)],
        ]
        for i in range(len(elo_stats[0])):
            window = extended_stats[i : i + window_size]
            means.append(mean(window))
        means[0] = 1000
        sns.set_theme()
        plt.figure(dpi=300)
        plt.bar(
            elo_stats[0],
            elo_stats[1],
            alpha=0.6,
            color=sns.color_palette("colorblind")[0],
            zorder=1,
        )
        plt.plot(
            elo_stats[0], means, color=sns.color_palette("colorblind")[0], zorder=2
        )
        plt.suptitle(
            f"Elo of agent {self.agent.__class__.__name__} during {self.count - 1} battles"
        )
        plt.title("from a new account")
        plt.xlabel("Battles")
        plt.ylabel("Elo")
        plt.ylim(0, max(elo_stats[1]) * 1.1)
        if len(elo_stats[0]) < 10:
            plt.gca().tick_params(axis="x", label1On=False)
        plt.tight_layout()
        plt.savefig(self.plot_path, backend="agg")

    def reset_elo(self):
        p = ResetProcess(self.username, self.password, self.battle_format)
        p.start()
        p.join()


class ResetPlayer(Player):
    def choose_move(
        self, battle: AbstractBattle
    ) -> Union[BattleOrder, Awaitable[BattleOrder]]:
        return ForfeitBattleOrder()


class ResetProcess(multiprocessing.Process):
    def __init__(self, username, password, battle_format, *args, **kwargs):
        self.username = username
        self.password = password
        self.battle_format = battle_format
        super().__init__(*args, **kwargs)

    def run(self) -> None:
        print(f"Starting reset process for player {self.username}...")
        player_config = PlayerConfiguration(self.username, self.password)
        agent = ResetPlayer(
            player_configuration=player_config,
            battle_format=self.battle_format,
            server_configuration=ShowdownServerConfiguration,
        )
        current_elo = get_ratings(self.username, self.battle_format)["elo"]
        print(f"Current elo for player {self.username}: {current_elo}")
        while current_elo != 1000:
            time.sleep(20)
            print(f"Starting battle to forfeit for player {self.username}...")
            asyncio.get_event_loop().run_until_complete(agent.ladder(1))
            print(f"Battle forfeited for player {self.username}...")
            new_elo = get_ratings(self.username, self.battle_format)["elo"]
            while new_elo == current_elo:
                time.sleep(1)
                new_elo = get_ratings(self.username, self.battle_format)["elo"]
            current_elo = new_elo
            print(f"Current elo for player {self.username}: {current_elo}")


if __name__ == "__main__":  # pragma: no cover
    multiprocessing.set_start_method("spawn")
    main()
