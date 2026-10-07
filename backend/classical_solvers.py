"""Step 3: classical layer. Assign orders to vehicles, then build / improve each route."""
import math
import numpy as np
from backend.vrp import route_cost


def sweep_assign(P):
    """Split customers into one group per vehicle by angle around the depot (capacity checked)."""
    nodes = list(range(1, P.n + 1))
    if P.n > P.params.n_vehicles * P.params.capacity:
        raise ValueError("Not enough vehicle capacity: increase n_vehicles or capacity.")
    nodes.sort(key=lambda i: math.atan2(P.lat[i] - P.lat[0], P.lng[i] - P.lng[0]))
    groups = [list(g) for g in np.array_split(nodes, P.params.n_vehicles)]
    groups = [[int(x) for x in g] for g in groups]
    for g in groups:
        if len(g) > P.params.capacity:
            raise ValueError("A group exceeds vehicle capacity: increase capacity or n_vehicles.")
    return groups


def fcfs_routes(groups, P):
    """Baseline: visit orders in the order they became ready (first come, first served)."""
    return [sorted(g, key=lambda i: P.ready[i]) for g in groups]


def nearest_neighbour(group, P):
    """Always go to the stop that can be served earliest."""
    left, cur, t, route = set(group), 0, 0.0, []
    while left:
        def start(j):
            return max(t + P.time[cur, j], P.ready[j])
        j = min(left, key=start)
        t = start(j) + P.params.service_min
        route.append(j)
        left.remove(j)
        cur = j
    return route


def two_opt(route, P):
    """Reverse segments while the cost keeps dropping."""
    best, best_c = list(route), route_cost(P, route)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                c = route_cost(P, cand)
                if c < best_c - 1e-9:
                    best, best_c, improved = cand, c, True
    return best


def solve_classical(P):
    groups = sweep_assign(P)
    fcfs = fcfs_routes(groups, P)
    nn = [nearest_neighbour(g, P) for g in groups]
    opt = [two_opt(r, P) for r in nn]
    return {"groups": groups, "fcfs": fcfs, "nearest_neighbour": nn, "two_opt": opt}