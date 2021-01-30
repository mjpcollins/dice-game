from agents.base_agent import DiceGameAgent


class OneStepLookAheadAgent(DiceGameAgent):

    def __init__(self, game):
        super().__init__(game)
        self._current_state = None
        self._risk_factor = 0.5

    def play(self, state):
        self._current_state = state
        return self._get_best_option()

    def _get_best_option(self):
        best_option = self._get_options_probs()[0]
        if best_option['prob_good_result'] >= self._risk_factor:
            return best_option['action']
        return tuple(i for i in range(len(self._current_state)))

    def _get_options_probs(self):
        all_options = [{"action": a, "prob_good_result":  self._calc_better_state_prob(action=a)}
                       for a in self.game.actions]
        all_options.sort(key=lambda x: x['prob_good_result'], reverse=True)
        return all_options

    def _calc_better_state_prob(self, action):
        prob_better_state = 0

        states, game_over, reward, probabilities = self.game.get_next_states(action=action,
                                                                             dice_state=self._current_state)
        if game_over:
            return prob_better_state

        for result_state, prob in zip(states, probabilities):
            if (self.game.final_scores[result_state] + reward) > self.game.final_scores[self._current_state]:
                prob_better_state += prob

        return prob_better_state
