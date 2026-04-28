"""
plots.py — Exercices 6, 7, 8
Tous les graphiques de comparaison des algorithmes DP.
"""

import random
import numpy as np
import matplotlib.pyplot as plt

import algorithms as alg
from game import ShortSnake, all_states, ACTIONS
from algorithms import (
    policy_evaluation, policy_evaluation_inplace,
    policy_improvement,
    policy_iteration_once, policy_iteration_once_inplace,
    value_iteration_once, value_iteration_rand,
    random_state,
)

N = 4
GAMMA = 0.9
THRESH = 0.01
N_INSTANCES = 100
N_STEPS = 1000

# Helper commun : mesure de performance d'une policy

def measure_performance(policy, n_instances=N_INSTANCES, n_steps=N_STEPS, n=N):
    """
    Lance n_instances parties de n_steps chaque.
    Retourne la récompense moyenne par 100 steps (slope * 100).
    """
    all_scores = []
    for _ in range(n_instances):
        snake = ShortSnake(policy, n)
        scores = []
        for _ in range(n_steps):
            snake.play_once()
            scores.append(snake.score)
        all_scores.append(scores)
    avg_scores = np.mean(all_scores, axis=0)
    slope, _ = np.polyfit(range(n_steps), avg_scores, 1)
    return slope * 100


def random_policy(n=N):
    return {s: random.choice(ACTIONS) for s in all_states(n)}

# Exercice 6 — Policy Iteration vs Value Iteration

def ex6_policy_iteration_sync(max_iter=12, n=N):
    """Q1-Q2 : Policy Iteration synchrone (v remis à 0 chaque fois)."""
    policy = random_policy(n)
    perfs = []
    slopes = []
    for i in range(max_iter):
        policy, v = policy_iteration_once(policy, None, GAMMA, THRESH, n)
        p = measure_performance(policy, n=n)
        perfs.append(p)
        slopes.append(p)
        print(f"[PI sync cold] iter {i+1:2d}  perf={p:.2f}")
    return perfs, slopes


def ex6_policy_iteration_sync_warm(max_iter=12, n=N):
    """Q3 : Policy Iteration synchrone avec warm start."""
    policy = random_policy(n)
    v = None
    perfs = []
    for i in range(max_iter):
        policy, v = policy_iteration_once(policy, v, GAMMA, THRESH, n)
        p = measure_performance(policy, n=n)
        perfs.append(p)
        print(f"[PI sync warm] iter {i+1:2d}  perf={p:.2f}")
    return perfs


def ex6_policy_iteration_inplace(max_iter=12, n=N):
    """Q4 : Policy Iteration en place avec warm start."""
    policy = random_policy(n)
    v = None
    perfs = []
    for i in range(max_iter):
        policy, v = policy_iteration_once_inplace(policy, v, GAMMA, THRESH, n)
        p = measure_performance(policy, n=n)
        perfs.append(p)
        print(f"[PI inplace]   iter {i+1:2d}  perf={p:.2f}")
    return perfs


def ex6_value_iteration(max_iter=50, n=N):
    """Q5 : Value Iteration (en place) + policy improvement pour extraire la policy."""
    v = {s: 0.0 for s in all_states(n)}
    perfs = []
    for i in range(max_iter):
        value_iteration_once(v, GAMMA, n)
        policy = policy_improvement(v, GAMMA, n)
        p = measure_performance(policy, n=n)
        perfs.append(p)
        print(f"[VI]           iter {i+1:2d}  perf={p:.2f}")
    return perfs


def ex6_plot_comparison(n=N):
    """Q6-Q8 : Compare toutes les méthodes en nb d'itérations et nb d'updates."""
    print("=== Ex6 — comparaison des méthodes ===")
    alg.update_count = 0; pi_sync_perfs, _ = ex6_policy_iteration_sync(n=n)
    pi_sync_updates = alg.update_count

    alg.update_count = 0; pi_warm_perfs = ex6_policy_iteration_sync_warm(n=n)
    pi_warm_updates = alg.update_count

    alg.update_count = 0; pi_inplace_perfs = ex6_policy_iteration_inplace(n=n)
    pi_inplace_updates = alg.update_count

    alg.update_count = 0; vi_perfs = ex6_value_iteration(n=n)
    vi_updates = alg.update_count

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot par itération
    ax = axes[0]
    ax.plot(pi_sync_perfs, label='PI sync (cold start)', marker='o')
    ax.plot(pi_warm_perfs, label='PI sync (warm start)', marker='s')
    ax.plot(pi_inplace_perfs, label='PI en place', marker='^')
    ax.plot(vi_perfs, label='Value Iteration', marker='x')
    ax.set_xlabel('Itération')
    ax.set_ylabel('Récompense / 100 steps')
    ax.set_title('Performance vs itérations')
    ax.legend()
    ax.grid(True)

    # Plot par nb d'updates — on construit des axes proportionnels
    ax = axes[1]
    n_states = len(list(all_states(n)))

    def updates_axis(perfs, total_updates):
        k = len(perfs)
        step = total_updates / k
        return [step * (i + 1) for i in range(k)]

    ax.plot(updates_axis(pi_sync_perfs, pi_sync_updates), pi_sync_perfs, label='PI sync (cold)', marker='o')
    ax.plot(updates_axis(pi_warm_perfs, pi_warm_updates), pi_warm_perfs, label='PI sync (warm)', marker='s')
    ax.plot(updates_axis(pi_inplace_perfs, pi_inplace_updates), pi_inplace_perfs, label='PI en place', marker='^')
    ax.plot(updates_axis(vi_perfs, vi_updates), vi_perfs, label='Value Iteration', marker='x')
    ax.set_xlabel('Nombre total d\'updates')
    ax.set_ylabel('Récompense / 100 steps')
    ax.set_title('Performance vs nb d\'updates')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('ex6_comparison.png', dpi=120)
    plt.show()
    print("Sauvegardé : ex6_comparison.png")


# Exercice 7 — GPI avec biais fréquence de visite

def ex7_gpi_biased(n_updates_list=None, max_cycles=200, n=N):
    """
    Pour plusieurs valeurs de n_updates, trace :
    - perf vs nb total d'updates
    - nb de cycles pour converger
    """
    if n_updates_list is None:
        n_updates_list = [10, 50, 100, 500]

    # 1. Initialiser visit_counts depuis 1000 steps random
    init_policy = random_policy(n)
    visit_counts = {s: 1 for s in all_states(n)}
    snake = ShortSnake(init_policy, n)
    for _ in range(N_STEPS):
        snake.play_once()
        visit_counts[snake.state] = visit_counts.get(snake.state, 1) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    conv_cycles = []

    for n_upd in n_updates_list:
        alg.update_count = 0
        v = {s: 0.0 for s in all_states(n)}
        policy = random_policy(n)
        perfs = []
        update_history = []
        prev_perf = None
        converged_at = None

        for cycle in range(max_cycles):
            value_iteration_rand(v, visit_counts, GAMMA, n_upd)
            policy = policy_improvement(v, GAMMA, n)
            p = measure_performance(policy, n_instances=30, n_steps=500, n=n)
            perfs.append(p)
            update_history.append(alg.update_count)

            if prev_perf is not None and abs(p - prev_perf) < 0.5 and converged_at is None:
                converged_at = cycle
            prev_perf = p

        conv_cycles.append(converged_at if converged_at is not None else max_cycles)
        axes[0].plot(update_history, perfs, label=f'n_upd={n_upd}')
        print(f"[GPI n_upd={n_upd}] convergence au cycle {converged_at}")

    axes[0].set_xlabel("Nombre total d'updates")
    axes[0].set_ylabel("Récompense / 100 steps")
    axes[0].set_title("GPI biaisé — perf vs updates")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(n_updates_list, conv_cycles, marker='o', color='tab:orange')
    axes[1].set_xlabel("n_updates par cycle")
    axes[1].set_ylabel("Cycles jusqu'à convergence")
    axes[1].set_title("Trade-off granularité / efficacité")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('ex7_gpi_biased.png', dpi=120)
    plt.show()
    print("Sauvegardé : ex7_gpi_biased.png")

# Exercice 8 — Différentes récompenses

class ShortSnakeCustomRewards(ShortSnake):
    """Version de ShortSnake avec récompenses configurables."""
    WALL_REWARD = -100
    FRUIT_REWARD = 10
    STEP_REWARD = 0

    @staticmethod
    def next_states(state, action):
        n = ShortSnake.GRID_SIZE
        hr, hc, fr, fc = state
        delta = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
        dr, dc = delta[action]
        new_hr, new_hc = hr + dr, hc + dc

        if not (0 <= new_hr < n and 0 <= new_hc < n):
            return [(ShortSnakeCustomRewards.WALL_REWARD, state)]

        if (new_hr, new_hc) == (fr, fc):
            transitions = []
            for nfr in range(n):
                for nfc in range(n):
                    if (nfr, nfc) != (new_hr, new_hc):
                        transitions.append((ShortSnakeCustomRewards.FRUIT_REWARD,
                                            (new_hr, new_hc, nfr, nfc)))
            return transitions

        return [(ShortSnakeCustomRewards.STEP_REWARD, (new_hr, new_hc, fr, fc))]


def run_value_iteration_custom(wall_r, fruit_r, step_r, max_iter=40, n=N):
    """Lance Value Iteration avec récompenses personnalisées, retourne policy + v."""
    ShortSnakeCustomRewards.WALL_REWARD = wall_r
    ShortSnakeCustomRewards.FRUIT_REWARD = fruit_r
    ShortSnakeCustomRewards.STEP_REWARD = step_r
    # Monkey-patch pour que next_states utilise les récompenses custom
    orig = ShortSnake.next_states
    ShortSnake.next_states = ShortSnakeCustomRewards.next_states

    v = {s: 0.0 for s in all_states(n)}
    for _ in range(max_iter):
        value_iteration_once(v, GAMMA, n)
    policy = policy_improvement(v, GAMMA, n)

    ShortSnake.next_states = orig
    return policy, v


def ex8_reward_analysis(n=N):
    """Q1-Q8 : Analyse des différentes récompenses."""
    configs = [
        ("−100/+10/0 (baseline)",  -100,  10,   0),
        ("−10/+1/0 (x0.1)",        -10,   1,    0),
        ("−100/+10/−1 (pénalité)", -100,  10,  -1),
        ("−5/+10/−1 (mur faible)", -5,    10,  -1),
    ]

    results = {}
    for label, w, f, s in configs:
        print(f"Running config: {label}")
        policy, v = run_value_iteration_custom(w, f, s, n=n)
        # Réinitialiser pour la mesure de perf (récompenses baseline)
        perf = measure_performance(policy, n_instances=50, n_steps=500, n=n)
        results[label] = {'policy': policy, 'v': v, 'perf': perf}
        print(f"  perf = {perf:.2f}")

    # Q1-Q2 : comparaison baseline vs x0.1
    baseline_v = results["−100/+10/0 (baseline)"]['v']
    scaled_v = results["−10/+1/0 (x0.1)"]['v']

    v_vals_base = [baseline_v[s] for s in all_states(n)]
    v_vals_scaled = [scaled_v[s] for s in all_states(n)]
    ratio = [b / (s * 10 + 1e-9) for b, s in zip(v_vals_base, v_vals_scaled)]
    print(f"\nQ1 — Ratio v_baseline / (v_scaled * 10) : mean={np.mean(ratio):.3f}, std={np.std(ratio):.3f}")

    # Policy identique ?
    p_base = results["−100/+10/0 (baseline)"]['policy']
    p_scaled = results["−10/+1/0 (x0.1)"]['policy']
    diff = sum(1 for s in all_states(n) if p_base[s] != p_scaled[s])
    print(f"Q2 — Nb états avec policy différente (baseline vs x0.1) : {diff} / {len(list(all_states(n)))}")

    # Plot des performances
    labels_plot = list(results.keys())
    perfs_plot = [results[l]['perf'] for l in labels_plot]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels_plot)), perfs_plot, color=['steelblue', 'orange', 'green', 'red'])
    plt.xticks(range(len(labels_plot)), labels_plot, rotation=15, ha='right')
    plt.ylabel('Récompense / 100 steps')
    plt.title('Ex8 — Impact des récompenses sur la performance')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('ex8_rewards.png', dpi=120)
    plt.show()
    print("Sauvegardé : ex8_rewards.png")

# Main

if __name__ == '__main__':
    print("=== Exercice 1 Q8-Q10 : Simulation de base ===")
    policy = {s: random.choice(ACTIONS) for s in all_states(N)}
    snake = ShortSnake(policy, N)
    scores_single = []
    for _ in range(N_STEPS):
        snake.play_once()
        scores_single.append(snake.score)

    all_scores_multi = []
    for _ in range(N_INSTANCES):
        p = {s: random.choice(ACTIONS) for s in all_states(N)}
        s = ShortSnake(p, N)
        scores = []
        for _ in range(N_STEPS):
            s.play_once()
            scores.append(s.score)
        all_scores_multi.append(scores)
    avg_scores = np.mean(all_scores_multi, axis=0)
    slope, _ = np.polyfit(range(N_STEPS), avg_scores, 1)
    print(f"Récompense moyenne / 100 steps (random policy) : {slope * 100:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(scores_single)
    axes[0].set_title("Q8 — Score d'une partie (policy random)")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Score cumulé")
    axes[1].plot(avg_scores)
    axes[1].set_title("Q9 — Score moyen (100 parties)")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Score moyen")
    plt.tight_layout()
    plt.savefig('ex1_simulation.png', dpi=120)
    plt.show()
    print("Sauvegardé : ex1_simulation.png")

    print("\n=== Exercice 6 ===")
    ex6_plot_comparison(n=N)

    print("\n=== Exercice 7 ===")
    ex7_gpi_biased(n_updates_list=[10, 50, 100, 500], n=N)

    print("\n=== Exercice 8 ===")
    ex8_reward_analysis(n=N)
