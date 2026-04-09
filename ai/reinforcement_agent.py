import numpy as np

class TradingAgent:

    def __init__(self):

        self.q_table = {}

    def choose_action(self, state):

        if state not in self.q_table:
            self.q_table[state] = [0,0,0]

        return np.argmax(self.q_table[state])