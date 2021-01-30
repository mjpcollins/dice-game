from agents.base_agent import DiceGameAgent


class MarkovDecisionProcessAgent(DiceGameAgent):

    def __init__(self, game, run_iterations=True, theta=0.001, gamma=0.95):
        super().__init__(game)
        self._gamma = gamma
        self._state_scores = {key: 0 for key in self.game.final_scores}
        self._changes = []
        self._largest_change = 100000.0
        self._state_best_action = {key: () for key in self.game.final_scores}
        self._iterations = 0
        self._theta = theta
        if run_iterations:
            self._iterate_until_minimal_change()

    def play(self, state):
        return self._state_best_action[state]

    def _iterate_until_minimal_change(self):
        while self._largest_change > self._theta:
            self._iterate_all_states()

    def _iterate_all_states(self):
        for state in self._state_scores:
            self._update_state_best_action(state)
        self._update_largest_change()
        self._iterations += 1

    def _update_state_best_action(self, state):
        best_action = ()
        best_action_value = -1000
        for action in self.game.actions:
            action_value = self._calculate_action_value(action=action,
                                                        state=state)
            if action_value > best_action_value:
                best_action = action
                best_action_value = action_value
        self._changes.append((self._state_scores[state] - best_action_value) ** 2)
        self._state_best_action[state] = best_action
        self._state_scores[state] = best_action_value

    def _calculate_action_value(self, action, state):
        states, game_over, reward, probabilities = self.game.get_next_states(action=action,
                                                                             dice_state=state)
        if game_over:
            return self.game.final_scores[state]
        expected_value = sum([self._state_scores[s] * p for p, s in zip(probabilities, states)])
        return reward + self._gamma * expected_value

    def _update_largest_change(self):
        self._largest_change = max(self._changes)
        self._changes = []

