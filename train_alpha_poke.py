from agents.alpha_poke import (
    AlphaPokeDeepDoubleDQN,
    AlphaPokeDeepSingleDQN,
    AlphaPokeDoubleDQN,
    AlphaPokeSingleDQN,
)


def train_single_dqn(
    steps: int,
    save_policy: str,
    battle_format: str = "gen8randombattle",
    logs: str = "./logs",
):
    step_factor = 10

    if steps % step_factor != 0:
        raise ValueError(
            f"Expected number of steps to be a multiple of 10. Got {steps}"
        )

    agent = AlphaPokeSingleDQN(
        battle_format=battle_format, eval_interval=50_000, log_interval=1000, test=True
    )
    agent.train(steps // 10)
    agent.save_policy(save_policy)
    agent.save_training_data(logs)


def train_double_dqn(
    steps: int,
    save_policy: str,
    battle_format: str = "gen8randombattle",
    logs: str = "./logs",
):
    step_factor = 10

    if steps % step_factor != 0:
        raise ValueError(
            f"Expected number of steps to be a multiple of 10. Got {steps}"
        )

    agent = AlphaPokeDoubleDQN(
        battle_format=battle_format, eval_interval=50_000, log_interval=1000, test=True
    )
    agent.train(steps // 10)
    agent.save_policy(save_policy)
    agent.save_training_data(logs)


def train_deep_single_dqn(
    steps: int,
    save_policy: str,
    battle_format: str = "gen8randombattle",
    logs: str = "./logs",
):
    step_factor = 10

    if steps % step_factor != 0:
        raise ValueError(
            f"Expected number of steps to be a multiple of 10. Got {steps}"
        )

    agent = AlphaPokeDeepSingleDQN(
        battle_format=battle_format, eval_interval=50_000, log_interval=1000, test=True
    )
    agent.train(steps // 10)
    agent.save_policy(save_policy)
    agent.save_training_data(logs)


def train_deep_double_dqn(
    steps: int,
    save_policy: str,
    battle_format: str = "gen8randombattle",
    logs: str = "./logs",
):
    step_factor = 10

    if steps % step_factor != 0:
        raise ValueError(
            f"Expected number of steps to be a multiple of 10. Got {steps}"
        )

    agent = AlphaPokeDeepDoubleDQN(
        battle_format=battle_format, eval_interval=50_000, log_interval=1000, test=True
    )
    agent.train(steps // 10)
    agent.save_policy(save_policy)
    agent.save_training_data(logs)


if __name__ == "__main__":
    iterations = int(input("Insert number of iterations: "))
    policy_path = input("Insert policy path: ")
    b_format = input("Insert battle format: ")
    log_folder = input("Insert log folder: ")
    choose_message = "\n\nChoose which agent to train:\n"
    choose_message += "1: DQN single battle\n"
    choose_message += "2: Double DQN single battle\n"
    choose_message += "3: Deep Single DQN single battle\n"
    choose_message += "4: Deep Double DQN single battle\n"
    choice = int(input(choose_message))
    if choice == 1:
        train_single_dqn(iterations, policy_path, b_format, log_folder)
    elif choice == 2:
        train_double_dqn(iterations, policy_path, b_format, log_folder)
    elif choice == 3:
        train_deep_single_dqn(iterations, policy_path, b_format, log_folder)
    elif choice == 4:
        train_deep_double_dqn(iterations, policy_path, b_format, log_folder)
    else:
        NotImplementedError(f"Choice {choice} not yet implemented.")
