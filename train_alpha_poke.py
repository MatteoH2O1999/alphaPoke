from agents.alpha_poke import AlphaPokeSingleDQN, AlphaPokeDoubleDQN


def train_single_dqn(
    steps: int,
    save_policy: str,
    battle_format: str = "gen8randombattle",
    logs: str = "./logs",
):
    agent = AlphaPokeSingleDQN(
        battle_format=battle_format, eval_interval=10_000, log_interval=1000
    )
    agent.train(steps)
    agent.save_policy(save_policy)
    agent.save_training_data(logs)


def train_double_dqn(
    steps: int,
    save_policy: str,
    battle_format: str = "gen8randombattle",
    logs: str = "./logs",
):
    agent = AlphaPokeDoubleDQN(
        battle_format=battle_format, eval_interval=10_000, log_interval=1000
    )
    agent.train(steps)
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
    choice = int(input(choose_message))
    if choice == 1:
        train_single_dqn(iterations, policy_path, b_format, log_folder)
    elif choice == 2:
        train_double_dqn(iterations, policy_path, b_format, log_folder)
    else:
        NotImplementedError(f"Choice {choice} not yet implemented.")
