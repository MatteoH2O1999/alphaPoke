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
from agents.advanced_heuristics import AdvancedHeuristics
from agents.alpha_poke import AlphaPokeSingleBattleModelLoader
from agents.dad import Dad
from agents.eight_year_old_me import EightYearOldMe
from agents.sarsa_stark import SarsaStark, ExpertSarsaStark
from agents.twenty_year_old_me import TwentyYearOldMe
from utils.close_player import close_player
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
    close_player(player)
