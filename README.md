# deepRL_tp4

---

## Partie Framework :

### Exercice 1 :

#### Question 1 :
- **Réponse :** L'état du jeu est représenté par un tuple `(head_row, head_col, fruit_row, fruit_col)`. On n'a pas besoin de la direction puisque le serpent ne grandit pas — il peut aller dans n'importe quelle direction depuis n'importe quel état.
- **Code :**
  ```python
  state = (head_row, head_col, fruit_row, fruit_col)
  ```

#### Question 2 :
- **Réponse :** Il y a $n^2 \times (n^2 - 1)$ états valides, car la tête et le fruit ne peuvent pas être sur la même case. Pour $n=4$ : $16 \times 15 = 240$ états.
- **Code :**
  ```python
  nb_etats = n**2 * (n**2 - 1)  # = 240 pour n=4
  ```

#### Question 3 :
- **Réponse :** On itère sur toutes les combinaisons `(head, fruit)` et on filtre celles où tête == fruit.
- **Code :**
  ```python
  def all_states(n):
      for hr in range(n):
          for hc in range(n):
              for fr in range(n):
                  for fc in range(n):
                      if (hr, hc) != (fr, fc):
                          yield (hr, hc, fr, fc)
  ```

#### Question 4 :
- **Réponse :** La policy aléatoire associe une action choisie uniformément parmi `['UP','DOWN','LEFT','RIGHT']` à chaque état.
- **Code :**
  ```python
  policy = {s: random.choice(ACTIONS) for s in all_states(n)}
  ```

#### Question 5 :
- **Réponse :** Le constructeur initialise `self.n`, `self.policy`, `self.state` (tiré aléatoirement parmi les états valides) et `self.score = 0`. `GRID_SIZE` est mis à jour comme attribut de classe pour que `next_states` y ait accès.
- **Code :**
  ```python
  def __init__(self, policy, n=4):
      self.n = n
      ShortSnake.GRID_SIZE = n
      self.policy = policy
      self.state = random.choice(list(all_states(n)))
      self.score = 0
  ```

#### Question 6 :
- **Réponse :** `next_states` implémente $p(s', r \mid s, a)$. Trois cas : (1) mur → récompense −100, état inchangé ; (2) fruit mangé → récompense +10, le fruit respawn aléatoirement sur une des $n^2-2$ cases libres ; (3) déplacement normal → récompense 0, nouvelle position de la tête.
- **Code :**
  ```python
  if not (0 <= new_hr < n and 0 <= new_hc < n):       # mur
      return [(-100, state)]
  if (new_hr, new_hc) == (fr, fc):                     # fruit mangé
      return [(10, (new_hr, new_hc, nfr, nfc))
              for nfr in range(n) for nfc in range(n)
              if (nfr, nfc) != (new_hr, new_hc)]
  return [(0, (new_hr, new_hc, fr, fc))]               # déplacement
  ```

#### Question 7 :
- **Réponse :** `play_once` récupère l'action de la policy, tire aléatoirement une transition (toutes équiprobables), met à jour l'état et le score.
- **Code :**
  ```python
  def play_once(self):
      action = self.policy[self.state]
      transitions = ShortSnake.next_states(self.state, action)
      reward, new_state = random.choice(transitions)
      self.state = new_state
      self.score += reward
      return action, reward
  ```

#### Question 8 :
- **Réponse :** Le score d'une seule instance fluctue beaucoup : avec une policy aléatoire, le serpent heurte souvent les murs (−100) et mange rarement le fruit (+10), donc le score descend globalement.

#### Question 9 :
- **Réponse :** La moyenne sur 100 instances lisse les fluctuations et révèle la tendance nette : avec une policy random le score moyen diminue progressivement car les pénalités de mur dominent.

#### Question 10 :
- **Réponse :** La pente de la régression linéaire multipliée par 100 donne la récompense moyenne par 100 steps. Avec une policy aléatoire cette valeur est négative (autour de −300 à −500 / 100 steps), confirmant que la policy random est sous-optimale.
- **Code :**
  ```python
  slope, _ = np.polyfit(range(1000), avg_scores, 1)
  reward_per_100 = slope * 100
  ```

---

## Partie Algorithmes DP :

### Exercice 2 — Policy Evaluation :

#### Question 1 :
- **Réponse :** On applique l'équation de Bellman pour un état : $v_{k+1}(s) = \sum_{s',r} p(s',r \mid s,\pi(s))[r + \gamma v_k(s')]$. Toutes les transitions sont équiprobables donc $p = 1/|\text{transitions}|$.
- **Code :**
  ```python
  def policy_evaluation_state(policy, v, gamma, state):
      action = policy[state]
      transitions = ShortSnake.next_states(state, action)
      p = 1.0 / len(transitions)
      return sum(p * (r + gamma * v.get(s2, 0)) for r, s2 in transitions)
  ```

#### Question 2 :
- **Réponse :** L'update synchrone calcule toutes les nouvelles valeurs depuis l'ancienne `v` (sans la modifier), puis les retourne dans `new_v`. `max_delta` mesure la convergence.
- **Code :**
  ```python
  def policy_evaluation_once(policy, v, gamma, n=4):
      new_v = {}
      max_delta = 0.0
      for state in all_states(n):
          new_val = policy_evaluation_state(policy, v, gamma, state)
          max_delta = max(max_delta, abs(new_val - v.get(state, 0.0)))
          new_v[state] = new_val
      return new_v, max_delta
  ```

#### Question 3 :
- **Réponse :** On boucle jusqu'à ce que `delta < thresh`, en remplaçant `v` par `new_v` à chaque itération synchrone.
- **Code :**
  ```python
  def policy_evaluation(policy, gamma, thresh=0.01, n=4):
      v = {s: 0.0 for s in all_states(n)}
      iterations = 0
      while True:
          v, delta = policy_evaluation_once(policy, v, gamma, n)
          iterations += 1
          if delta < thresh:
              break
      return v, iterations
  ```

#### Question 4 :
- **Réponse :** La version en place modifie `v[state]` directement à chaque état sans attendre la fin du sweep. Cela converge souvent plus vite car les nouvelles valeurs sont immédiatement réutilisées dans le même sweep.
- **Code :**
  ```python
  def policy_evaluation_once_inplace(policy, v, gamma, n=4):
      max_delta = 0.0
      for state in all_states(n):
          old_val = v[state]
          v[state] = policy_evaluation_state(policy, v, gamma, state)
          max_delta = max(max_delta, abs(v[state] - old_val))
      return max_delta
  ```

#### Question 5 :
- **Réponse :** Avec une policy aléatoire, le serpent va souvent dans les murs (récompense −100), donc la value function des états proches des bords est très négative. $v(s) \ll 0$ pour les états où la policy aléatoire choisit fréquemment une action menant à un mur.

---

### Exercice 3 — Policy Improvement et Iteration :

#### Question 1 :
- **Réponse :** `best_action` calcule $Q(s,a)$ pour chaque action et retourne celle qui maximise $Q$.
- **Code :**
  ```python
  def best_action(state, v, gamma):
      return max(ACTIONS, key=lambda a: q_value(state, a, v, gamma))
  ```

#### Question 2 :
- **Réponse :** `policy_improvement` applique l'amélioration greedy à tous les états : $\pi'(s) = \arg\max_a Q(s,a,v_\pi)$.
- **Code :**
  ```python
  def policy_improvement(v, gamma, n=4):
      return {s: best_action(s, v, gamma) for s in all_states(n)}
  ```

#### Question 3 :
- **Réponse :** Un cycle de policy iteration évalue la policy courante jusqu'à convergence, puis l'améliore. On peut passer `v=None` pour réinitialiser ou passer la `v` précédente pour le warm start.
- **Code :**
  ```python
  def policy_iteration_once(policy, v, gamma, thresh=0.01, n=4):
      if v is None:
          v = {s: 0.0 for s in all_states(n)}
      v, _ = policy_evaluation(policy, gamma, thresh, n)
      new_policy = policy_improvement(v, gamma, n)
      return new_policy, v
  ```

#### Question 4 :
- **Réponse :** La version en place utilise `policy_evaluation_inplace` — même logique, convergence généralement plus rapide (moins de sweeps nécessaires par évaluation).

---

### Exercice 4 — Value Iteration :

#### Question 1 :
- **Réponse :** `value_update_state` prend le $\max$ sur toutes les actions au lieu de suivre $\pi(s)$ : $v_{k+1}(s) = \max_a \sum p(s',r|s,a)[r + \gamma v_k(s')]$.
- **Code :**
  ```python
  def value_update_state(v, gamma, state):
      return max(q_value(state, a, v, gamma) for a in ACTIONS)
  ```

#### Question 2 :
- **Réponse :** `value_iteration_once` fait un sweep en place sur tous les états : pour chaque état, on met à jour `v[state]` immédiatement (asynchrone), et on track `max_delta`.
- **Code :**
  ```python
  def value_iteration_once(v, gamma, n=4):
      max_delta = 0.0
      for state in all_states(n):
          old_val = v[state]
          v[state] = value_update_state(v, gamma, state)
          max_delta = max(max_delta, abs(v[state] - old_val))
      return max_delta
  ```

---

### Exercice 5 — Biaiser l'ordre d'évaluation :

#### Question 1 :
- **Réponse :** On initialise tous les états à 1 (pas 0) pour éviter qu'un état non visité ait une probabilité nulle d'être choisi.
- **Code :**
  ```python
  visit_counts = {s: 1 for s in all_states(n)}
  ```

#### Question 2 :
- **Réponse :** `random_state` tire un état proportionnellement à sa fréquence de visite via `random.choices` avec poids.
- **Code :**
  ```python
  def random_state(visit_counts):
      states = list(visit_counts.keys())
      weights = [visit_counts[s] for s in states]
      return random.choices(states, weights=weights, k=1)[0]
  ```

#### Question 3 :
- **Réponse :** `value_iteration_rand` fait `n_updates` mises à jour en place, états tirés selon `visit_counts`. Cela concentre les updates sur les états les plus visités, ce qui est plus efficace en pratique.
- **Code :**
  ```python
  def value_iteration_rand(v, visit_counts, gamma, n_updates=10):
      for _ in range(n_updates):
          state = random_state(visit_counts)
          v[state] = value_update_state(v, gamma, state)
  ```

---

## Partie Graphiques :

### Exercice 6 — Policy Iteration vs Value Iteration :

#### Question 1-2 :
- **Réponse :** Policy Iteration synchrone avec cold start (v=0 à chaque itération) converge en quelques itérations mais chaque itération est coûteuse car `policy_evaluation` fait de nombreux sweeps internes. La courbe perf vs itérations monte rapidement puis se stabilise.

#### Question 3 :
- **Réponse :** Avec warm start (réutilisation du `v` précédent), `policy_evaluation` nécessite beaucoup moins de sweeps car `v` est déjà proche de la solution. La convergence est donc plus rapide en nombre total d'itérations.

#### Question 4 :
- **Réponse :** La version en place converge encore plus vite (moins de sweeps par évaluation) car les nouvelles valeurs sont immédiatement disponibles pour les états suivants du même sweep.

#### Question 5 :
- **Réponse :** Value Iteration fait un seul sweep par itération (pas d'évaluation complète). La policy est extraite à la fin par `policy_improvement`. Chaque "itération" est bien moins coûteuse.

#### Question 6 :
- **Réponse :** En nombre d'itérations, Policy Iteration semble converger en moins d'itérations. Mais en nombre total d'updates, Value Iteration est souvent plus efficace : elle atteint de bonnes performances avec beaucoup moins d'updates totaux.

#### Question 7-8 :
- **Réponse :** En retraçant en fonction du compteur global `update_count`, Value Iteration apparaît nettement plus efficace que Policy Iteration avec évaluation complète. Les versions en place et warm start réduisent l'écart mais Value Iteration reste compétitive.

---

### Exercice 7 — GPI avec biais fréquence de visite :

#### Question 1-3 :
- **Réponse :** On initialise `visit_counts` en jouant 1000 steps avec une policy aléatoire, puis on alterne `value_iteration_rand` (updates biaisés) et `policy_improvement`. La courbe perf vs updates montre une convergence progressive vers la policy optimale.

#### Question 4-5 :
- **Réponse :** Pour `n_updates` petit (10), on fait beaucoup de cycles de policy improvement — overhead élevé. Pour `n_updates` grand (500), chaque cycle est long mais on fait peu d'appels à `policy_improvement`. Le coude optimal se trouve généralement autour de `n_updates` = 50-100 pour n=4 : bon compromis entre granularité des updates et fréquence d'amélioration de la policy.

---

### Exercice 8 — Différentes récompenses :

#### Question 1 :
- **Réponse :** Les configs −100/+10/0 et −10/+1/0 ont le même ratio récompenses. La value function de la seconde est environ 10× plus petite que celle de la première, car toutes les valeurs sont multipliées par le même facteur.

#### Question 2 :
- **Réponse :** La policy est identique (0 états différents) : les deux configurations définissent le même MDP à une constante multiplicative près, donc le $\arg\max$ des Q-values reste identique.

#### Question 3-4 :
- **Réponse :** Avec −100/+10/−1, chaque pas coûte −1. La policy devient plus directe : le serpent évite les détours inutiles. Les routes longues qui n'étaient pas pénalisées disparaissent de la policy optimale.

#### Question 5-6 :
- **Réponse :** Avec −5/+10/−1, la punition mur est faible. Pour des états très proches d'un mur avec le fruit de l'autre côté, traverser le mur (−5) peut être plus rentable que de faire le tour (coût = nb_steps × 1). La policy change pour ces états : l'agent peut choisir de foncer dans le mur.

#### Question 7-8 :
- **Réponse :** La punition minimale rendant la traversée non rentable est telle que :

$$|r_{\text{mur}}| \geq d_{\text{detour}}$$

  Pour un détour de 2 steps (cas d'un coin), le seuil est −2. On peut le trouver par dichotomie sur `wall_reward` : chercher le `x` tel qu'avec `wall = x−ε` la policy traverse le mur, et avec `wall = x` elle contourne. L'état limite est typiquement un état en bord de grille avec le fruit adjacent de l'autre côté du mur.
