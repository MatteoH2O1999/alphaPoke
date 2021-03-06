from .action_space_init import init_action_space
from .action_to_move_function import get_int_action_to_move, get_int_action_space_size
from .invalid_argument import InvalidArgument, InvalidArgumentNumber


def argmax(iterable):
    indexes = list(range(len(iterable)))
    return max(indexes, key=lambda x: iterable[x])
