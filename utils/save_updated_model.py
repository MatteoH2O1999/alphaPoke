#
# A pokémon showdown battle-bot project based on reinforcement learning techniques.
# Copyright (C) 2022 Matteo Dell'Acqua
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Update the agent model after training
import datetime
import os
import pickle

from agents.base_classes.trainable_player import TrainablePlayer


def update_model(agent: TrainablePlayer, model_path):
    current_time = datetime.datetime.now()
    current_time_string = current_time.strftime("%d-%m-%Y %H-%M-%S")
    agent_name = agent.__class__.__name__
    save_path = model_path
    if agent_name == "SimpleRLAgent":
        folder_name = "simpleRL"
    elif agent_name == "ExpertRLAgent":
        folder_name = "expertRL"
    elif agent_name == "SarsaStark":
        folder_name = "SarsaStark"
    elif agent_name == "ExpertSarsaStark":
        folder_name = "expertSarsaStark"
    else:
        raise RuntimeError(f"{agent_name} is not a valid trainable agent")
    save_path = os.path.join(
        save_path, folder_name, agent.b_format, f"updated {current_time_string}.pokeai"
    )
    with open(save_path, "wb") as model_file:
        pickle.dump(agent.get_model(), model_file)
