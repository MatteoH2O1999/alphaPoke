from agents.advanced_heuristics import AdvancedHeuristics
from agents.alpha_poke import AlphaPokeSingleBattleModelLoader
from agents.dad import Dad
from agents.eight_year_old_me import EightYearOldMe
from agents.sarsa_stark import SarsaStark, ExpertSarsaStark
from agents.twenty_year_old_me import TwentyYearOldMe
from gui_data import PLAYER_TYPE_DICT


AGENT_CLASSES = {
    "dad": Dad,
    "8-year-old-me": EightYearOldMe,
    "20-year-old-me": TwentyYearOldMe,
    "advanced-heuristics": AdvancedHeuristics,
    "simpleSarsaStark-best": SarsaStark,
    "expertSarsaStark-best": ExpertSarsaStark,
}

for cli_name, _, _ in PLAYER_TYPE_DICT.values():
    if "alphaPokeSingle-" in cli_name:
        AGENT_CLASSES[cli_name] = AlphaPokeSingleBattleModelLoader


def agents():
    player_list = []
    for player_type in PLAYER_TYPE_DICT.values():
        cli_name = player_type[0]
        available_battle_formats = list(player_type[2].values())
        expected_class = AGENT_CLASSES[cli_name]
        for battle_format in available_battle_formats:
            player_list.append((cli_name, expected_class, battle_format))
    return player_list
