# Allows bots to play on the ladder
import asyncio
import datetime
import getpass
import matplotlib.pyplot as plt
import multiprocessing
import os
import seaborn as sns
import sys

from poke_env.server_configuration import ShowdownServerConfiguration
from poke_env.player_configuration import PlayerConfiguration

from agents.trainable_player import TrainablePlayer
from utils.create_agent import create_agent
from utils.save_updated_model import update_model


def main():
    current_time = datetime.datetime.now()
    current_time_string = current_time.strftime('%d-%m-%Y %H-%M-%S')
    battle_format = sys.argv[1]
    save_replays = sys.argv[2].lower() == 'true' or sys.argv[2].lower() == 't' or sys.argv[2].lower() == 'y'
    save_path = sys.argv[3]
    processes = []
    max_completed_battles = multiprocessing.Value('i', 0)
    cont = multiprocessing.Value('i', 1)
    for i in range(4, len(sys.argv)):
        processes.append(PlayerProcess(sys.argv[i], battle_format, save_replays, save_path, current_time_string,
                                       max_completed_battles, cont))
    for p in processes:
        p.start()
    input('Press any key to stop...')
    with cont.get_lock():
        cont.value = 0
    for p in processes:
        p.join()


class PlayerProcess(multiprocessing.Process):

    def __init__(self, agent_type, battle_format, save_replays, save_path, plot_time, stop, cont, **kwargs):
        super().__init__(**kwargs)
        self.cont = True
        self.cont_int = cont
        self.stop_on_shared = stop
        self.username = input('Account username: ')
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
        plt.switch_backend('agg')
        self.agent = create_agent(self.agent_type, PlayerConfiguration(self.username, self.password),
                                  self.battle_format, True, ShowdownServerConfiguration, self.save_replays)[0]
        file_name = f'rank {self.agent.__class__.__name__} {self.plot_time}.png'
        self.plot_path = os.path.join(self.save_path, file_name)
        elo_stats = [[0], [0]]
        while self.cont or self.stop_on >= self.count:
            asyncio.get_event_loop().run_until_complete(self.agent.ladder(1))
            elo_stats[0].append(self.count)
            battles = self.agent.battles
            if len(battles) != 1:
                raise RuntimeError('???')
            for b in battles.values():
                if b.rating:
                    elo_stats[1].append(b.rating)
                else:
                    elo_stats[1].append(elo_stats[1][-1])
            self.agent.reset_battles()
            with self.stop_on_shared.get_lock():
                if self.stop_on_shared.value < self.count:
                    self.stop_on_shared.value = self.count
                self.stop_on = self.stop_on_shared.value
            with self.cont_int.get_lock():
                if self.cont_int.value == 1:
                    self.cont = True
                elif self.cont_int.value == 0:
                    self.cont = False
                else:
                    raise RuntimeError(f'Invalid value. Expected 0 or 1, got {self.cont_int.value}')
            self.count += 1
        if isinstance(self.agent, TrainablePlayer) and (self.agent.training or self.agent.train_while_playing):
            update_model(self.agent, './models')
        if len(elo_stats[1]) > 1:
            elo_stats[0] = elo_stats[0][1:]
            elo_stats[1] = elo_stats[1][1:]
        os.makedirs(os.path.dirname(self.plot_path), exist_ok=True)
        sns.set_theme()
        plt.figure(dpi=300)
        plt.plot(elo_stats[0], elo_stats[1], color=sns.color_palette('colorblind')[0])
        plt.suptitle(f'Rank of agent {self.agent.__class__.__name__} during {self.count - 1} battles')
        plt.title('from a new account')
        plt.xlabel('Battles')
        plt.ylabel('Elo')
        plt.ylim(0, max(elo_stats[1]) * 1.1)
        plt.gca().tick_params(axis='x', label1On=False)
        plt.savefig(self.plot_path, backend='agg')


if __name__ == '__main__':
    main()
