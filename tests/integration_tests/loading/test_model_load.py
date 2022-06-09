from agents.dad import Dad
from agents.eight_year_old_me import EightYearOldMe
from agents.sarsa_stark import SarsaStark, ExpertSarsaStark
from agents.seba import Seba
from agents.twenty_year_old_me import TwentyYearOldMe
from utils.close_player import close_player
from utils.create_agent import create_agent


def test_load_dad():
    player = create_agent("dad", battle_format="gen8randombattle")[0]
    assert isinstance(player, Dad)
    assert player.format == "gen8randombattle"
    assert not player.format_is_doubles
    assert player.logged_in
    close_player(player)


def test_load_8_year_old_me():
    player = create_agent("8-year-old-me", battle_format="gen8randombattle")[0]
    assert isinstance(player, EightYearOldMe)
    assert player.format == "gen8randombattle"
    assert not player.format_is_doubles
    assert player.logged_in
    close_player(player)


def test_load_20_year_old_me():
    player = create_agent("20-year-old-me", battle_format="gen8randombattle")[0]
    assert isinstance(player, TwentyYearOldMe)
    assert player.format == "gen8randombattle"
    assert not player.format_is_doubles
    assert player.logged_in
    close_player(player)


def test_load_seba():
    player = create_agent("seba", battle_format="gen8randombattle")[0]
    assert isinstance(player, Seba)
    assert player.format == "gen8randombattle"
    assert not player.format_is_doubles
    assert player.logged_in
    close_player(player)


def test_load_simple_sarsa():
    player = create_agent("simpleSarsaStark-best", battle_format="gen8randombattle")[0]
    assert isinstance(player, SarsaStark)
    assert player.format == "gen8randombattle"
    assert not player.format_is_doubles
    assert player.logged_in
    assert player.model != {}
    close_player(player)


def test_load_expert_sarsa():
    player = create_agent("expertSarsaStark-best", battle_format="gen8randombattle")[0]
    assert isinstance(player, ExpertSarsaStark)
    assert player.format == "gen8randombattle"
    assert not player.format_is_doubles
    assert player.logged_in
    assert player.model != {}
    close_player(player)
