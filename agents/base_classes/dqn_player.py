# Base class for a DQN Player
from abc import ABC, abstractmethod
from tf_agents.agents.tf_agent import LossInfo

from agents.base_classes.tf_player import TFPlayer


class DQNPlayer(TFPlayer, ABC):
    def train(self, num_iterations: int):
        print("Creating train step counter...")
        self.agent.train_step_counter.assign(0)
        print("Evaluating initial policy...")
        self.eval_function(self.agent.train_step_counter.numpy())
        if (
            self.wrapped_env.challenge_task is None
            or self.wrapped_env.challenge_task.done()
        ):
            print("Starting challenge loop...")
            self.wrapped_env.start_challenging()
        print("Collecting samples with random policy...")
        self.random_driver.run(self.environment.reset())
        print("Resetting the environment...")
        time_step = self.environment.reset()
        print("Training...")
        for _ in range(num_iterations):
            time_step, _ = self.collect_driver.run(time_step)
            experience, unused_info = next(self.replay_buffer_iterator)
            loss_data = self.agent.train(experience)
            step = self.agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                self.log_function(step, loss_data)

            if step % self.eval_interval == 0:
                self.wrapped_env.close(purge=False)
                self.eval_function(step)
                self.wrapped_env.start_challenging()
                time_step = self.environment.reset()

    @abstractmethod
    def eval_function(self, step):  # pragma: no cover
        pass

    @abstractmethod
    def log_function(self, step, loss_info: LossInfo):  # pragma: no cover
        pass
