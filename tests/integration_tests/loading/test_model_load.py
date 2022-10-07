from agents.advanced_heuristics import AdvancedHeuristics
from agents.alpha_poke import AlphaPokeSingleBattleModelLoader
from agents.dad import Dad
from agents.eight_year_old_me import EightYearOldMe
from agents.sarsa_stark import SarsaStark, ExpertSarsaStark
from agents.twenty_year_old_me import TwentyYearOldMe
from utils.create_agent import create_agent


def test_load_dad():
    player = create_agent("dad", battle_format="gen8randombattle")[0]
    assert isinstance(player, Dad)
    assert player.format == "gen8randombattle"
    assert not player.format_is_doubles


def test_load_8_year_old_me():
    player = create_agent("8-year-old-me", battle_format="gen8randombattle")[0]
    assert isinstance(player, EightYearOldMe)
    assert player.format == "gen8randombattle"
    assert not player.format_is_doubles


def test_load_20_year_old_me():
    player = create_agent("20-year-old-me", battle_format="gen8randombattle")[0]
    assert isinstance(player, TwentyYearOldMe)
    assert player.format == "gen8randombattle"
    assert not player.format_is_doubles


def test_load_advanced_heuristics():
    player = create_agent("advanced-heuristics", battle_format="gen8randombattle")[0]
    assert isinstance(player, AdvancedHeuristics)
    assert player.format == "gen8randombattle"
    assert not player.format_is_doubles


def test_load_simple_sarsa():
    player = create_agent("simpleSarsaStark-best", battle_format="gen8randombattle")[0]
    assert isinstance(player, SarsaStark)
    assert player.format == "gen8randombattle"
    assert not player.format_is_doubles
    assert player.model != {}


def test_load_expert_sarsa():
    player = create_agent("expertSarsaStark-best", battle_format="gen8randombattle")[0]
    assert isinstance(player, ExpertSarsaStark)
    assert player.format == "gen8randombattle"
    assert not player.format_is_doubles
    assert player.model != {}


def test_load_alpha_poke_single():
    player = create_agent(
        "alphaPokeSingle-doubleDQNsingle/simple-embedding",
        battle_format="gen8randombattle",
    )[0]
    assert isinstance(player, AlphaPokeSingleBattleModelLoader)
    assert player.format == "gen8randombattle"
    assert not player.format_is_doubles
    assert not player.can_train
