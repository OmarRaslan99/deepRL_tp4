import random

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
GRID_SIZE = 4  # default, overridable


def all_states(n=GRID_SIZE):
    """Générateur de tous les états valides (head != fruit)."""
    for hr in range(n):
        for hc in range(n):
            for fr in range(n):
                for fc in range(n):
                    if (hr, hc) != (fr, fc):
                        yield (hr, hc, fr, fc)


class ShortSnake:
    GRID_SIZE = GRID_SIZE

    def __init__(self, policy, n=GRID_SIZE):
        self.n = n
        ShortSnake.GRID_SIZE = n
        self.policy = policy
        # Choisir un état initial aléatoire valide
        states = list(all_states(n))
        self.state = random.choice(states)
        self.score = 0

    @staticmethod
    def next_states(state, action):
        """
        Calcule p(s', r | s, a) et retourne une liste de (reward, next_state).
        - Mur        : [(−100, state)]
        - Fruit mangé: [(+10, (new_h_r, new_h_c, nfr, nfc)) for chaque case vide]
        - Déplacement: [(0, (new_hr, new_hc, fr, fc))]
        """
        n = ShortSnake.GRID_SIZE
        hr, hc, fr, fc = state

        delta = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
        dr, dc = delta[action]
        new_hr, new_hc = hr + dr, hc + dc

        # Cas mur
        if not (0 <= new_hr < n and 0 <= new_hc < n):
            return [(-100, state)]

        # Cas fruit mangé
        if (new_hr, new_hc) == (fr, fc):
            transitions = []
            for nfr in range(n):
                for nfc in range(n):
                    if (nfr, nfc) != (new_hr, new_hc):
                        transitions.append((10, (new_hr, new_hc, nfr, nfc)))
            return transitions

        # Cas déplacement normal
        return [(0, (new_hr, new_hc, fr, fc))]

    def play_once(self):
        """Joue un step selon la policy courante."""
        action = self.policy[self.state]
        transitions = ShortSnake.next_states(self.state, action)
        reward, new_state = random.choice(transitions)
        self.state = new_state
        self.score += reward
        return action, reward
