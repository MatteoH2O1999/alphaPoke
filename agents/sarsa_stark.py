# AI trained with SARSA or Q-learning algorithm
from .basic_rl import SimpleRLAgent
from .expert_rl import ExpertRLAgent


class SarsaStark(SimpleRLAgent):

    def _train(self, last_state, last_action, reward):
        # TODO
        pass


class ExpertSarsaStark(ExpertRLAgent):

    def _train(self, last_state, last_action, reward):
        # TODO
        pass


class QSarsaStark(SimpleRLAgent):

    def _train(self, last_state, last_action, reward):
        # TODO
        pass


class QExpertSarsaStark(ExpertRLAgent):

    def _train(self, last_state, last_action, reward):
        # TODO
        pass
