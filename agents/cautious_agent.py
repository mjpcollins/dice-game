from agents.one_step_look_ahead import OneStepLookAheadAgent


class CautiousAgent(OneStepLookAheadAgent):

    def __init__(self, game):
        super().__init__(game)
        self._risk_factor = 0.6
