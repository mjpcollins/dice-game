from agents.base_agent import DiceGameAgent


class MarkovDecisionProcessAgentAdjusted(DiceGameAgent):

    def __init__(self, game, run_iterations=True, theta=0.001, gamma=0.95):
        super().__init__(game)
        self._theta_squared = theta ** 2
        self._deltas_squared = []
        self._gamma = gamma
        self._state_action_value = {state: 0 for state in self.game.states}
        self._state_best_action = {state: () for state in self.game.states}
        self._next_states_dict = {state: {} for state in self.game.states}
        self._get_all_next_states()
        self._iterations = 0
        if run_iterations:
            self._iterate_until_minimal_delta()

    def play(self, state):
        return self._state_best_action[state]

    def _iterate_until_minimal_delta(self):
        self._iterate_all_states()
        while max(self._deltas_squared) > self._theta_squared:
            self._iterate_all_states()

    def _iterate_all_states(self):
        self._deltas_squared = []
        for state in self._state_action_value:
            self._update_state_best_action(state)
        self._iterations += 1

    def _update_state_best_action(self, state):
        action_values = [self._calculate_action_value(action=action, state=state) for action in self.game.actions]
        best_action_value = max(action_values)
        self._deltas_squared.append((self._state_action_value[state] - best_action_value) ** 2)
        self._state_action_value[state] = best_action_value
        self._state_best_action[state] = self.game.actions[action_values.index(best_action_value)]

    def _calculate_action_value(self, action, state):
        states, game_over, reward, probabilities = self._next_states_dict[state][action]
        if game_over:
            return self.game.final_scores[state]
        expected_value = sum([self._state_action_value[s] * p for p, s in zip(probabilities, states)])
        return reward + self._gamma * expected_value

    def _get_all_next_states(self):
        for state in self.game.states:
            for action in self.game.actions:
                self._next_states_dict[state][action] = self.game.get_next_states(action=action,
                                                                                  dice_state=state)
