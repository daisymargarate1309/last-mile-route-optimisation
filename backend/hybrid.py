"""Step 7: hybrid solver.
Classical: build routes (nearest neighbour) and judge every candidate with the exact simulator.
Quantum:   slide a window of k consecutive stops along each route; QAOA re-orders the window.
A QAOA answer is accepted ONLY if the exact route cost really improves."""
import itertools
import time
import numpy as np
from backend.vrp import route_cost
from backend.qubo_window import build_window_qubo, tour_to_index
from backend.qaoa_engine import QAOA
from backend.interference import path_analysis, label


def solve_window(P, route, s, k, reps, maxiter, restarts, seed):
    stops = route[s:s + k]
    prefix = route[:s]
    nxt = route[s + k] if s + k < len(route) else 0
    Q = build_window_qubo(P, prefix, stops, nxt)
    eng = QAOA(Q, reps=reps)
    opt = eng.optimize(maxiter=maxiter, restarts=restarts, seed=seed)
    psi, phi, layers = eng.run(eng.gammas, eng.betas)

    perms = list(itertools.permutations(range(k)))
    idxs = [tour_to_index(p, k) for p in perms]
    probs = np.abs(psi) ** 2

    def full_cost(p):
        return route_cost(P, prefix + [stops[i] for i in p] + route[s + k:])

    exact = {p: full_cost(p) for p in perms}
    exact_best = min(perms, key=lambda p: exact[p])
    ranked = sorted(zip(perms, idxs), key=lambda pi: -probs[pi[1]])
    qaoa_top = ranked[0][0]
    candidates = [p for p, _ in ranked[:3]]                 # 3 most probable tours -> exact check
    chosen = min(candidates, key=lambda p: exact[p])

    tours = []
    for p, ix in zip(perms, idxs):
        info = path_analysis(eng, phi, eng.betas[-1], ix)
        tours.append({
            "tour": [P.ids[stops[i]] for i in p],
            "exact_cost": round(exact[p], 2),
            "p_uniform": 1.0 / eng.N,
            "p_layers": [float(abs(l[ix]) ** 2) for l in layers],     # p0 (uniform), after layer 1, 2, ...
            "p_final": float(probs[ix]),
            "factor": round(info["factor"], 3),
            "coherence": round(info["coherence"], 3),            "phase": round(info["phase"], 3),
            "type": label(info["factor"]),
            "phase_hist": info["phase_hist"],
        })
    return {
        "chosen": [stops[i] for i in chosen],
        "improved": exact[chosen] < route_cost(P, route) - 1e-9,
        "qaoa_top_is_optimal": qaoa_top == exact_best,
        "qaoa_top_rank": sorted(exact.values()).index(exact[qaoa_top]),   # 0 = best tour
        "chosen_is_optimal": chosen == exact_best,
        "qubits": eng.n,
        "valid_prob_layers": [float(sum(abs(l[ix]) ** 2 for ix in idxs)) for l in layers],
        "valid_prob_uniform": len(idxs) / eng.N,
        "history": opt["history"],
        "gammas": [float(g) for g in eng.gammas],
        "betas": [float(b) for b in eng.betas],
        "tours": tours,
        "engine": eng,
    }

def window_starts(length, k):
    """Short routes: overlapping windows. Long routes: windows side by side (keeps run time bounded)."""
    if length <= 6:
        return list(range(0, length - k + 1))
    starts = list(range(0, length - k + 1, k))
    if starts[-1] != length - k:
        starts.append(length - k)
    return starts


def hybrid_solve(P, start_routes, k=3, reps=3, maxiter=150, restarts=3, seed=7, verbose=True,
                 time_limit=None, max_windows=None, info=None):
    routes = [list(r) for r in start_routes]
    logs = []
    t0 = time.time()
    active = [r for r in routes if len(r) >= 2]
    per_route = None
    if max_windows and active:
        per_route = max(1, max_windows // len(active))        # share the window budget between vehicles
    for v, route in enumerate(routes):
        kk = min(k, len(route))
        if kk < 2:
            continue
        starts = window_starts(len(route), kk)
        if per_route:
            starts = starts[:per_route]
        for s in starts:
            if time_limit and time.time() - t0 > time_limit:
                if info is not None:
                    info["truncated"] = True
                return routes, logs
            w = solve_window(P, route, s, kk, reps, maxiter, restarts, seed + 31 * v + s)
            before = route_cost(P, route)
            if w["improved"]:
                route[s:s + kk] = w["chosen"]
            w.update(vehicle=v + 1, start=s, cost_before=round(before, 2),
                     cost_after=round(route_cost(P, route), 2))
            logs.append(w)
            if verbose:
                print(f"  vehicle {v+1} window@{s}: {w['qubits']} qubits, P(valid) "
                      f"{w['valid_prob_uniform']:.3f} -> {w['valid_prob_layers'][-1]:.3f}, "
                      f"cost {w['cost_before']} -> {w['cost_after']}, "
                      f"QAOA top-1 optimal: {w['qaoa_top_is_optimal']}")
    return routes, logs
