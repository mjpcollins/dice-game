import numpy as np
from unittest import TestCase
from agents import OneStepLookAheadAgent
from dice_game import DiceGame


class TestOneStepLookAheadAgent(TestCase):

    def setUp(self):
        np.random.seed(1)
        self.game = DiceGame()
        self.agent = OneStepLookAheadAgent(self.game)
        self.agent._current_state = (1, 1, 2)
        self.state = self.game.reset()
