DAD_DESCRIPTION = "dad description"
EIGHT_YEAR_OLD_ME_DESCRIPTION = "8 year old me description"
TWENTY_YEAR_OLD_ME_DESCRIPTION = "20 year old me description"
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
    "Dad": (
        "dad",
        DAD_DESCRIPTION,
        DAD_SUPPORTED_BATTLES,
    ),
    "8-Year-Old me": (
        "8-year-old-me",
        EIGHT_YEAR_OLD_ME_DESCRIPTION,
        EIGHT_YEAR_OLD_ME_SUPPORTED_BATTLES,
    ),
    "20-Year-Old me": (
        "20-year-old-me",
        TWENTY_YEAR_OLD_ME_DESCRIPTION,
        TWENTY_YEAR_OLD_ME_SUPPORTED_BATTLES,
    ),
    "Simple Reinforcement Learning Player": (
        "simpleRL-best",
        SIMPLE_RL_DESCRIPTION,
        SIMPLE_RL_SUPPORTED_BATTLES,
    ),
    "Expert Reinforcement Learning Player": (
        "expertRL-best",
        EXPERT_RL_DESCRIPTION,
        EXPERT_RL_SUPPORTED_BATTLES,
    ),
    "Sarsa Trained Player": (
        "simpleSarsaStark-best",
        SIMPLE_SARSA_DESCRIPTION,
        SIMPLE_SARSA_SUPPORTED_BATTLES,
    ),
    "Expert Sarsa Trained Player": (
        "expertSarsaStark-best",
        EXPERT_SARSA_DESCRIPTION,
        EXPERT_SARSA_SUPPORTED_BATTLES,
    ),
}
