# Update the agent model after training
import datetime
import os
import pickle

from agents.trainable_player import TrainablePlayer


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
