from .action_space_init import init_action_space
from .invalid_argument import InvalidArgument, InvalidArgumentNumber


def argmax(iterable):
    indexes = list(range(len(iterable)))
    return max(indexes, key=lambda x: iterable[x])
