# Base class for a DQN Player
from abc import ABC, abstractmethod
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.drivers import py_driver
from tf_agents.policies import random_tf_policy, py_tf_eager_policy

from agents.base_classes.tf_player import TFPlayer


class DQNPlayer(TFPlayer, ABC):
    def train(self, num_iterations: int):
        self.agent.train_step_counter.assign(0)
        self.evaluations["returns"] = []
        self.eval_function(self.agent.train_step_counter.numpy())
        if (
            self.wrapped_env.challenge_task is None
            or self.wrapped_env.challenge_task.done()
        ):
            self.wrapped_env.start_challenging()
        random_policy = random_tf_policy.RandomTFPolicy(
            self.environment.time_step_spec(), self.environment.action_spec()
        )
        py_driver.PyDriver(
            self.environment,
            py_tf_eager_policy.PyTFEagerPolicy(
                random_policy, use_tf_function=True, batch_time_steps=False
            ),
            [self.replay_buffer.add_batch],
            max_steps=100,
        ).run(self.environment.reset())
        time_step = self.environment.reset()
        for _ in range(num_iterations):
            time_step, _ = self.collect_driver.run(time_step)
            experience, unused_info = next(self.replay_buffer_iterator)
            loss_data = self.agent.train(experience)
            step = self.agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                self.log_function(step, loss_data)

            if step % self.eval_interval == 0:
                self.eval_function(step)

    @abstractmethod
    def eval_function(self, step):
        pass

    @abstractmethod
    def log_function(self, step, loss_info: LossInfo):
        pass
