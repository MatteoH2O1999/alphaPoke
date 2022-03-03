import unittest.mock

from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.player.random_player import RandomPlayer

from utils.action_to_move_function import action_to_move_gen8single


def get_mocks():
    mock_agent = unittest.mock.create_autospec(
        RandomPlayer, spec_set=True, instance=True
    )
    battle = Battle("battle_tag", "username", None)  # noqa
    return mock_agent, battle


# Generation 8 single battle tests


def test_forfeit():
    mock_agent, battle = get_mocks()
    assert isinstance(
        action_to_move_gen8single(mock_agent, -1, battle), ForfeitBattleOrder
    )


def test_gen8single_choose_normal_move_success():
    mock_agent, battle = get_mocks()
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 1, battle)
    mock_agent.create_order.assert_called_with(moves[1])


def test_gen8single_choose_normal_move_failure_out_of_range():
    mock_agent, battle = get_mocks()
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 2, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


def test_gen8single_choose_normal_move_failure_forced_switch():
    mock_agent, battle = get_mocks()
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    battle._force_switch = True
    action_to_move_gen8single(mock_agent, 1, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


@unittest.mock.patch(
    "poke_env.environment.pokemon.Pokemon.available_z_moves",
    new_callable=unittest.mock.PropertyMock,
)
def test_gen8single_choose_z_move_success(available_z_moves_mock):
    mock_agent, battle = get_mocks()
    battle._can_z_move = True
    active_pokemon = Pokemon(species="charizard")
    active_pokemon._active = True
    battle._team = {"charizerd": active_pokemon}
    z_move = Move("flamethrower")
    available_z_moves_mock.return_value = [z_move]
    action_to_move_gen8single(mock_agent, 4, battle)
    mock_agent.create_order.assert_called_with(z_move, z_move=True)


@unittest.mock.patch(
    "poke_env.environment.pokemon.Pokemon.available_z_moves",
    new_callable=unittest.mock.PropertyMock,
)
def test_gen8single_choose_z_move_failure_forced_switch(available_z_moves_mock):
    mock_agent, battle = get_mocks()
    battle._can_z_move = True
    battle._force_switch = True
    active_pokemon = Pokemon(species="charizard")
    active_pokemon._active = True
    battle._team = {"charizerd": active_pokemon}
    z_move = Move("flamethrower")
    available_z_moves_mock.return_value = [z_move]
    action_to_move_gen8single(mock_agent, 4, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


@unittest.mock.patch(
    "poke_env.environment.pokemon.Pokemon.available_z_moves",
    new_callable=unittest.mock.PropertyMock,
)
def test_gen8single_choose_z_move_failure_cannot_z_move(available_z_moves_mock):
    mock_agent, battle = get_mocks()
    battle._can_z_move = False
    active_pokemon = Pokemon(species="charizard")
    active_pokemon._active = True
    battle._team = {"charizerd": active_pokemon}
    z_move = Move("flamethrower")
    available_z_moves_mock.return_value = [z_move]
    action_to_move_gen8single(mock_agent, 4, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


@unittest.mock.patch(
    "poke_env.environment.pokemon.Pokemon.available_z_moves",
    new_callable=unittest.mock.PropertyMock,
)
def test_gen8single_choose_z_move_failure_no_active_pokemon(available_z_moves_mock):
    mock_agent, battle = get_mocks()
    battle._can_z_move = True
    active_pokemon = Pokemon(species="charizard")
    active_pokemon._active = False
    battle._team = {"charizerd": active_pokemon}
    z_move = Move("flamethrower")
    available_z_moves_mock.return_value = [z_move]
    action_to_move_gen8single(mock_agent, 4, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


@unittest.mock.patch(
    "poke_env.environment.pokemon.Pokemon.available_z_moves",
    new_callable=unittest.mock.PropertyMock,
)
def test_gen8single_choose_z_move_failure_out_of_range(available_z_moves_mock):
    mock_agent, battle = get_mocks()
    battle._can_z_move = True
    active_pokemon = Pokemon(species="charizard")
    active_pokemon._active = True
    battle._team = {"charizerd": active_pokemon}
    z_move = Move("flamethrower")
    available_z_moves_mock.return_value = [z_move]
    action_to_move_gen8single(mock_agent, 5, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


def test_gen8single_choose_mega_move_success():
    mock_agent, battle = get_mocks()
    battle._can_mega_evolve = True
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 9, battle)
    mock_agent.create_order.assert_called_with(moves[1], mega=True)


def test_gen8single_choose_mega_move_failure_cannot_mega_evolve():
    mock_agent, battle = get_mocks()
    battle._can_mega_evolve = False
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 9, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


def test_gen8single_choose_mega_move_failure_forced_switch():
    mock_agent, battle = get_mocks()
    battle._can_mega_evolve = True
    battle._force_switch = True
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 9, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


def test_gen8single_choose_mega_move_failure_out_of_range():
    mock_agent, battle = get_mocks()
    battle._can_mega_evolve = True
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 10, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


def test_gen8single_choose_dynamax_move_success():
    mock_agent, battle = get_mocks()
    battle._can_dynamax = True
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 13, battle)
    mock_agent.create_order.assert_called_with(moves[1], dynamax=True)


def test_gen8single_choose_dynamax_move_failure_cannot_dynamax():
    mock_agent, battle = get_mocks()
    battle._can_dynamax = False
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 13, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


def test_gen8single_choose_dynamax_move_failure_forced_switch():
    mock_agent, battle = get_mocks()
    battle._can_dynamax = True
    battle._force_switch = True
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 13, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


def test_gen8single_choose_dynamax_move_failure_out_of_range():
    mock_agent, battle = get_mocks()
    battle._can_dynamax = True
    moves = [Move("flamethrower"), Move("tackle")]
    battle._available_moves = moves
    action_to_move_gen8single(mock_agent, 14, battle)
    mock_agent.choose_random_move.assert_called_with(battle)


def test_gen8single_switch_success():
    mock_agent, battle = get_mocks()
    switches = [Pokemon(species="charizard"), Pokemon(species="pikachu")]
    battle._available_switches = switches
    action_to_move_gen8single(mock_agent, 17, battle)
    mock_agent.create_order.assert_called_with(switches[1])


def test_gen8single_switch_failure():
    mock_agent, battle = get_mocks()
    switches = [Pokemon(species="charizard"), Pokemon(species="pikachu")]
    battle._available_switches = switches
    action_to_move_gen8single(mock_agent, 18, battle)
    mock_agent.choose_random_move.assert_called_with(battle)
