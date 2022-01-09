from .create_agent import create_agent
from .action_space_init import init_action_space


def argmax(iterable):
    indexes = list(range(len(iterable)))
    return max(indexes, key=lambda x: iterable[x])
