import random
from game import ShortSnake, all_states, ACTIONS

# Compteur global d'updates (réinitialisé manuellement selon le besoin)
update_count = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def q_value(state, action, v, gamma):
    """Q(s, a) = Σ p(s',r|s,a) [r + γ v(s')]"""
    transitions = ShortSnake.next_states(state, action)
    p = 1.0 / len(transitions)
    return sum(p * (r + gamma * v.get(s2, 0.0)) for r, s2 in transitions)


def best_action(state, v, gamma):
    """Retourne l'action greedy vis-à-vis de v."""
    return max(ACTIONS, key=lambda a: q_value(state, a, v, gamma))


# ---------------------------------------------------------------------------
# Exercice 2 — Policy Evaluation
# ---------------------------------------------------------------------------

def policy_evaluation_state(policy, v, gamma, state):
    """
    Équation de Bellman pour un état :
    v_{k+1}(s) = Σ p(s',r|s,π(s)) [r + γ v_k(s')]
    """
    action = policy[state]
    transitions = ShortSnake.next_states(state, action)
    p = 1.0 / len(transitions)
    return sum(p * (r + gamma * v.get(s2, 0.0)) for r, s2 in transitions)


def policy_evaluation_once(policy, v, gamma, n=4):
    """Update synchrone : calcule new_v depuis l'ancienne v."""
    global update_count
    new_v = {}
    max_delta = 0.0
    for state in all_states(n):
        new_val = policy_evaluation_state(policy, v, gamma, state)
        max_delta = max(max_delta, abs(new_val - v.get(state, 0.0)))
        new_v[state] = new_val
        update_count += 1
    return new_v, max_delta


def policy_evaluation(policy, gamma, thresh=0.01, n=4):
    """Boucle synchrone jusqu'à convergence. Retourne (v, iterations)."""
    v = {s: 0.0 for s in all_states(n)}
    iterations = 0
    while True:
        v, delta = policy_evaluation_once(policy, v, gamma, n)
        iterations += 1
        if delta < thresh:
            break
    return v, iterations


def policy_evaluation_once_inplace(policy, v, gamma, n=4):
    """Update en place (asynchrone) : modifie v directement."""
    global update_count
    max_delta = 0.0
    for state in all_states(n):
        old_val = v[state]
        v[state] = policy_evaluation_state(policy, v, gamma, state)
        max_delta = max(max_delta, abs(v[state] - old_val))
        update_count += 1
    return max_delta


def policy_evaluation_inplace(policy, gamma, thresh=0.01, n=4):
    """Boucle en place jusqu'à convergence. Retourne iterations."""
    v = {s: 0.0 for s in all_states(n)}
    iterations = 0
    while True:
        delta = policy_evaluation_once_inplace(policy, v, gamma, n)
        iterations += 1
        if delta < thresh:
            break
    return v, iterations


# ---------------------------------------------------------------------------
# Exercice 3 — Policy Improvement & Policy Iteration
# ---------------------------------------------------------------------------

def policy_improvement(v, gamma, n=4):
    """π'(s) = argmax_a Q(s, a, v) pour tout s."""
    return {s: best_action(s, v, gamma) for s in all_states(n)}


def policy_iteration_once(policy, v, gamma, thresh=0.01, n=4):
    """
    Un cycle Policy Iteration (synchrone).
    Si v is None → initialise à 0.
    Retourne (new_policy, v).
    """
    if v is None:
        v = {s: 0.0 for s in all_states(n)}
    v, _ = policy_evaluation(policy, gamma, thresh, n)
    new_policy = policy_improvement(v, gamma, n)
    return new_policy, v


def policy_iteration_once_inplace(policy, v, gamma, thresh=0.01, n=4):
    """
    Un cycle Policy Iteration (en place).
    Si v is None → initialise à 0.
    Retourne (new_policy, v).
    """
    if v is None:
        v = {s: 0.0 for s in all_states(n)}
    v, _ = policy_evaluation_inplace(policy, gamma, thresh, n)
    new_policy = policy_improvement(v, gamma, n)
    return new_policy, v


# ---------------------------------------------------------------------------
# Exercice 4 — Value Iteration
# ---------------------------------------------------------------------------

def value_update_state(v, gamma, state):
    """
    v_{k+1}(s) = max_a Σ p(s',r|s,a) [r + γ v(s')]
    """
    return max(q_value(state, a, v, gamma) for a in ACTIONS)


def value_iteration_once(v, gamma, n=4):
    """
    Un sweep en place sur tous les états.
    Retourne max_delta.
    """
    global update_count
    max_delta = 0.0
    for state in all_states(n):
        old_val = v[state]
        v[state] = value_update_state(v, gamma, state)
        max_delta = max(max_delta, abs(v[state] - old_val))
        update_count += 1
    return max_delta


# ---------------------------------------------------------------------------
# Exercice 5 — Biaiser l'ordre d'évaluation par fréquence de visite
# ---------------------------------------------------------------------------

def random_state(visit_counts):
    """Tire un état proportionnellement à sa fréquence de visite."""
    states = list(visit_counts.keys())
    weights = [visit_counts[s] for s in states]
    return random.choices(states, weights=weights, k=1)[0]


def value_iteration_rand(v, visit_counts, gamma, n_updates=10):
    """
    Effectue n_updates mises à jour en place, états tirés selon visit_counts.
    Met à jour update_count.
    """
    global update_count
    for _ in range(n_updates):
        state = random_state(visit_counts)
        old_val = v[state]
        v[state] = value_update_state(v, gamma, state)
        update_count += 1
