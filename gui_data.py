DAD_DESCRIPTION = (
    "A random player that chooses every action randomly from all the available ones. "
    "A player of this type is the weakest baseline without actively trying to lose."
)
EIGHT_YEAR_OLD_ME_DESCRIPTION = (
    "A player that uses the move with the highest base power, completely ignoring "
    "typing and additional effects. If it cannot attack, it will perform a random action "
    "(including switches)."
)
TWENTY_YEAR_OLD_ME_DESCRIPTION = (
    "A player that implements a basic strategy using simple heuristics."
)
SIMPLE_RL_DESCRIPTION = "simple rl description"
EXPERT_RL_DESCRIPTION = "expert rl description"
SIMPLE_SARSA_DESCRIPTION = "simple sarsa description"
EXPERT_SARSA_DESCRIPTION = "expert sarsa description"

DAD_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}
EIGHT_YEAR_OLD_ME_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}
TWENTY_YEAR_OLD_ME_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}
SIMPLE_RL_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}
EXPERT_RL_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}
SIMPLE_SARSA_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}
EXPERT_SARSA_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}

PLAYER_TYPE_DICT = {
    "Random Player": (
        "dad",
        DAD_DESCRIPTION,
        DAD_SUPPORTED_BATTLES,
    ),
    "Max Base Power Player": (
        "8-year-old-me",
        EIGHT_YEAR_OLD_ME_DESCRIPTION,
        EIGHT_YEAR_OLD_ME_SUPPORTED_BATTLES,
    ),
    "Simple Heuristics Player": (
        "20-year-old-me",
        TWENTY_YEAR_OLD_ME_DESCRIPTION,
        TWENTY_YEAR_OLD_ME_SUPPORTED_BATTLES,
    ),
}
