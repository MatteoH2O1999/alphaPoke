from agents.dad import Dad
from agents.eight_year_old_me import EightYearOldMe
from agents.sarsa_stark import SarsaStark, ExpertSarsaStark
from agents.advanced_heuristics import AdvancedHeuristics
from agents.twenty_year_old_me import TwentyYearOldMe


def agents():
    return [
        ("dad", Dad, "gen8randombattle"),
        ("8-year-old-me", EightYearOldMe, "gen8randombattle"),
        ("20-year-old-me", TwentyYearOldMe, "gen8randombattle"),
        ("advanced-heuristics", AdvancedHeuristics, "gen8randombattle"),
        ("simpleSarsaStark-best", SarsaStark, "gen8randombattle"),
        ("expertSarsaStark-best", ExpertSarsaStark, "gen8randombattle"),
    ]
