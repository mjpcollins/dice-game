import numpy as np
from unittest import TestCase
from agents import MarkovDecisionProcessAgentAdjusted
from dice_game import DiceGame


class TestMarkovDecisionProcessAgentAdjusted(TestCase):

    def setUp(self):
        np.random.seed(1)
        self.game = DiceGame()
        self.agent = MarkovDecisionProcessAgentAdjusted(self.game,
                                                        run_iterations=False)
        self.state = self.game.reset()

    def test_init(self):
        self.assertEqual(0.95, self.agent._gamma)
        self.assertEqual(1e-06, self.agent._theta_squared)
        self.assertEqual({(1, 1, 1): 0, (1, 1, 2): 0, (1, 1, 3): 0, (1, 1, 4): 0, (1, 1, 5): 0, (1, 1, 6): 0, (1, 2, 2): 0, (1, 2, 3): 0, (1, 2, 4): 0, (1, 2, 5): 0, (1, 2, 6): 0, (1, 3, 3): 0, (1, 3, 4): 0, (1, 3, 5): 0, (1, 3, 6): 0, (1, 4, 4): 0, (1, 4, 5): 0, (1, 4, 6): 0, (1, 5, 5): 0, (1, 5, 6): 0, (1, 6, 6): 0, (2, 2, 2): 0, (2, 2, 3): 0, (2, 2, 4): 0, (2, 2, 5): 0, (2, 2, 6): 0, (2, 3, 3): 0, (2, 3, 4): 0, (2, 3, 5): 0, (2, 3, 6): 0, (2, 4, 4): 0, (2, 4, 5): 0, (2, 4, 6): 0, (2, 5, 5): 0, (2, 5, 6): 0, (2, 6, 6): 0, (3, 3, 3): 0, (3, 3, 4): 0, (3, 3, 5): 0, (3, 3, 6): 0, (3, 4, 4): 0, (3, 4, 5): 0, (3, 4, 6): 0, (3, 5, 5): 0, (3, 5, 6): 0, (3, 6, 6): 0, (4, 4, 4): 0, (4, 4, 5): 0, (4, 4, 6): 0, (4, 5, 5): 0, (4, 5, 6): 0, (4, 6, 6): 0, (5, 5, 5): 0, (5, 5, 6): 0, (5, 6, 6): 0, (6, 6, 6): 0},
                         self.agent._state_action_value)
        self.assertEqual([], self.agent._deltas_squared)
        self.assertEqual({(1, 1, 1): (), (1, 1, 2): (), (1, 1, 3): (), (1, 1, 4): (), (1, 1, 5): (), (1, 1, 6): (), (1, 2, 2): (), (1, 2, 3): (), (1, 2, 4): (), (1, 2, 5): (), (1, 2, 6): (), (1, 3, 3): (), (1, 3, 4): (), (1, 3, 5): (), (1, 3, 6): (), (1, 4, 4): (), (1, 4, 5): (), (1, 4, 6): (), (1, 5, 5): (), (1, 5, 6): (), (1, 6, 6): (), (2, 2, 2): (), (2, 2, 3): (), (2, 2, 4): (), (2, 2, 5): (), (2, 2, 6): (), (2, 3, 3): (), (2, 3, 4): (), (2, 3, 5): (), (2, 3, 6): (), (2, 4, 4): (), (2, 4, 5): (), (2, 4, 6): (), (2, 5, 5): (), (2, 5, 6): (), (2, 6, 6): (), (3, 3, 3): (), (3, 3, 4): (), (3, 3, 5): (), (3, 3, 6): (), (3, 4, 4): (), (3, 4, 5): (), (3, 4, 6): (), (3, 5, 5): (), (3, 5, 6): (), (3, 6, 6): (), (4, 4, 4): (), (4, 4, 5): (), (4, 4, 6): (), (4, 5, 5): (), (4, 5, 6): (), (4, 6, 6): (), (5, 5, 5): (), (5, 5, 6): (), (5, 6, 6): (), (6, 6, 6): ()},
                         self.agent._state_best_action)

    def test_calculate_action_value_hold_all(self):
        self.assertEqual(14, self.agent._calculate_action_value(action=(0, 1, 2),
                                                                state=(1, 1, 2)))

    def test_calculate_action_value_hold_0_and_1(self):
        actual_value = round(self.agent._calculate_action_value(action=(0, 1),
                                                                state=(1, 1, 2)), 4)
        self.assertEqual(-1, actual_value)

    def test_update_state_best_action(self):
        self.agent._gamma = 0.9
        self.agent._update_state_best_action(state=(1, 1, 2))
        self.assertEqual((0, 1, 2), self.agent._state_best_action[(1, 1, 2)])
        self.assertEqual(14, self.agent._state_action_value[(1, 1, 2)])

    def test_iterate_all_states_once(self):
        self.agent._iterate_all_states()
        self.assertEqual(324, max(self.agent._deltas_squared))
        self.assertEqual(1, self.agent._iterations)

    def test_iterate_all_states_5_times(self):
        for _ in range(5):
            self.agent._iterate_all_states()
        self.assertEqual(0.013878447483153854, max(self.agent._deltas_squared))
        self.assertEqual(5, self.agent._iterations)

    def test_iterate_until_minimal_change_01(self):
        self.agent._theta_squared = 0.1 ** 2
        self.agent._iterate_until_minimal_delta()
        self.assertEqual(0.002008600926252003, max(self.agent._deltas_squared))
        self.assertEqual(6, self.agent._iterations)

    def test_iterate_until_minimal_change_0001(self):
        self.agent._theta_squared = 0.001 ** 2
        self.agent._iterate_until_minimal_delta()
        self.assertEqual(1.6804616828459174e-07, max(self.agent._deltas_squared))
        self.assertEqual(11, self.agent._iterations)

    def test_init_with_iterations(self):
        agent = MarkovDecisionProcessAgentAdjusted(self.game)
        self.assertEqual(1.6804616828459174e-07, max(agent._deltas_squared))
        self.assertEqual(11, agent._iterations)

    def test_play(self):
        agent = MarkovDecisionProcessAgentAdjusted(self.game)
        self.assertEqual((0, 1), agent.play((1, 1, 2)))
        self.assertEqual((0, ), agent.play((1, 3, 5)))
        self.assertEqual((0, 1, 2), agent.play((4, 4, 6)))
        self.assertEqual((0, 1, 2), agent.play((2, 2, 4)))
