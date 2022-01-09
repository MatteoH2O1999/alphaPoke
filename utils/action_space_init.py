# Function to initialize the action space for RL custom algorithms
from typing import List


def init_action_space(length) -> List[int]:
    action_space = []
    for _ in range(length):
        action_space.append(0)
    return action_space
