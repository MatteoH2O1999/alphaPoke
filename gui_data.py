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
SIMPLE_SARSA_DESCRIPTION = (
    "A reinforcement learning player trained with SARSA algorithm. "
    "Its performance is similar to a Random Player."
)
EXPERT_SARSA_DESCRIPTION = (
    "A reinforcement learning player trained with SARSA algorithm "
    "and using a higher level action space ('Use supereffective move' "
    "instead of 'Use move 4'). Its performance is similar to a Max Base Power Player."
)
DOUBLE_DQN_SINGLE_BATTLE_1024_DESCRIPTION = (
    "A player trained on a dense neural network with 1024 neurons with ELU activation function. "
    "Uses a low level action space. "
    "Its performance is similar to a Max Base Power Player."
)

DAD_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}
EIGHT_YEAR_OLD_ME_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}
TWENTY_YEAR_OLD_ME_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}
SIMPLE_RL_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}
EXPERT_RL_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}
SIMPLE_SARSA_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}
EXPERT_SARSA_SUPPORTED_BATTLES = {"[Gen 8] Random Battle": "gen8randombattle"}
DOUBLE_DQN_SINGLE_BATTLE_1024_SUPPORTED_BATTLES = {
    "[Gen 8] Random Battle": "gen8randombattle"
}

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
    "SARSA Trained Player": (
        "simpleSarsaStark-best",
        SIMPLE_SARSA_DESCRIPTION,
        SIMPLE_SARSA_SUPPORTED_BATTLES,
    ),
    "SARSA Trained Expert Player": (
        "expertSarsaStark-best",
        EXPERT_SARSA_DESCRIPTION,
        EXPERT_SARSA_SUPPORTED_BATTLES,
    ),
    "Double DQN Neural-Network-1024 Player": (
        "alphaPokeSingle-doubleDQNsingle/simple-embedding",
        DOUBLE_DQN_SINGLE_BATTLE_1024_DESCRIPTION,
        DOUBLE_DQN_SINGLE_BATTLE_1024_SUPPORTED_BATTLES,
    ),
}
