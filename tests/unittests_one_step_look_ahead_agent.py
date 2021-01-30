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

    def test_calc_probs_hold_0_and_1(self):
        expected_prob = round(2/3, 4)
        actual_prob = self.agent._calc_better_state_prob(action=(0, 1))
        self.assertEqual(expected_prob, round(actual_prob, 4))

    def test_calc_probs_hold_all(self):
        actual_prob = self.agent._calc_better_state_prob(action=(0, 1, 2))
        self.assertEqual(0, actual_prob)

    def test_get_options_probs(self):
        expected_option_for_0_1 = round(2/3, 4)
        actual_options = self.agent._get_options_probs()
        self.assertEqual(8, len(actual_options))
        self.assertEqual(expected_option_for_0_1, round(actual_options[0]['prob_good_result'], 4))

    def test_get_best_option(self):
        expected_option = (0, 1)
        actual_option = self.agent._get_best_option()
        self.assertEqual(expected_option, actual_option)

    def test_get_best_option_uncertain(self):
        self.agent._current_state = (1, 3, 5)
        expected_option = ()
        actual_option = self.agent._get_best_option()
        self.assertEqual(expected_option, actual_option)

    def test_get_best_option_hold(self):
        self.agent._current_state = (1, 1, 6)
        expected_option = (0, 1, 2)
        actual_option = self.agent._get_best_option()
        self.assertEqual(expected_option, actual_option)