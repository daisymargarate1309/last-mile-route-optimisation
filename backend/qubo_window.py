"""Step 4: QUBO for ONE window of consecutive stops (k stops -> k*k qubits)."""
import numpy as np


def time_before_window(P, prefix, stops):
    """Return (time we leave `prev`, prev node) where prev = last stop before the window."""
    if not prefix:
        t = max(0.0, min(P.ready[s] - P.time[0, s] for s in stops))
        return t, 0
    t = max(0.0, P.ready[prefix[0]] - P.time[0, prefix[0]])
    prev = 0
    for c in prefix:
        t += P.time[prev, c]
        t = max(t, P.ready[c]) + P.params.service_min
        prev = c
    return t, prev


def _one_hot(Q, vars_, A):
    """Adds A * (sum(x) - 1)^2 (constant dropped)."""
    for a in vars_:
        Q[a, a] -= A
    for p in range(len(vars_)):
        for q in range(p + 1, len(vars_)):
            a, b = sorted((vars_[p], vars_[q]))
            Q[a, b] += 2 * A


def build_window_qubo(P, prefix, stops, nxt, penalty=3.0):
    """x[i,t] = 1  <=>  stop i is served at position t of the window.
    cost = travel cost (prev -> window -> next) + estimated lateness + one-hot penalties."""
    k = len(stops)
    n = k * k
    idx = lambda i, t: i * k + t
    t_dep, prev = time_before_window(P, prefix, stops)
    cpk, lp, svc = P.params.cost_per_km, P.params.late_penalty, P.params.service_min
    w = lambda a, b: cpk * P.dist[a, b]

    Q = np.zeros((n, n))
    for i, s in enumerate(stops):                       # prev -> first, last -> next
        Q[idx(i, 0), idx(i, 0)] += w(prev, s)
        Q[idx(i, k - 1), idx(i, k - 1)] += w(s, nxt)
    for t in range(k - 1):                              # consecutive stops
        for i in range(k):
            for j in range(k):
                if i != j:
                    a, b = sorted((idx(i, t), idx(j, t + 1)))
                    Q[a, b] += w(stops[i], stops[j])

    # time windows (linear approximation): estimated arrival of stop i at position t
    legs = [P.time[a, b] for a in stops for b in stops if a != b]
    avg_leg = float(np.mean(legs))
    first = float(np.mean([P.time[prev, s] for s in stops]))
    for i, s in enumerate(stops):
        for t in range(k):
            arrival = t_dep + first + t * (svc + avg_leg)
            Q[idx(i, t), idx(i, t)] += lp * max(0.0, arrival - P.due[s])

    Q /= max(np.abs(Q).max(), 1e-9)                     # normalise so penalty = 3 is meaningful
    for i in range(k):                                  # each stop exactly once
        _one_hot(Q, [idx(i, t) for t in range(k)], penalty)
    for t in range(k):                                  # each position exactly one stop
        _one_hot(Q, [idx(i, t) for i in range(k)], penalty)
    return Q


def tour_to_index(perm_local, k):
    """perm_local[t] = local stop index served at position t  ->  bitstring index."""
    return sum(1 << (i * k + t) for t, i in enumerate(perm_local))