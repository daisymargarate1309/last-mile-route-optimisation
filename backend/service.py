"""Step 8: one function that runs the whole pipeline and returns JSON-friendly data for the website."""
import math
import time
import numpy as np
from backend.lade_data import select_orders
from backend.vrp import Params, TRAFFIC, build_problem, evaluate_solution
from backend.classical_solvers import solve_classical
from backend.hybrid import hybrid_solve

TIME_LIMIT_S = 170          # stop starting new QAOA windows after this many seconds


def clean(o):
    """numpy -> plain python so it can be sent as JSON."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


def _int(v, lo, hi, default):
    try:
        v = int(float(v))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def run_optimization(mode="pickup", n_orders=20, n_vehicles=4, capacity=8, traffic="normal",
                     k=3, reps=3, seed=42, radius_km=6.0):
    # any value is accepted: out-of-range numbers are clamped, impossible combinations are fixed
    mode = mode if mode in ("pickup", "delivery") else "pickup"
    traffic = traffic if traffic in TRAFFIC else "normal"
    n_orders = _int(n_orders, 8, 40, 20)
    capacity = _int(capacity, 2, 20, 8)
    n_vehicles = _int(n_vehicles, 1, 20, 4)
    k = 4 if _int(k, 3, 4, 3) == 4 else 3
    reps = _int(reps, 1, 4, 3)
    seed = _int(seed, 0, 2 ** 31 - 1, 42)

    orders = select_orders(mode=mode, n=n_orders, seed=seed, radius_km=radius_km)
    n = len(orders)
    if n < 4:
        raise ValueError("Not enough orders found for this data / seed. Try another seed.")
    n_vehicles = min(max(n_vehicles, math.ceil(n / capacity)), n)     # enough vehicles for the capacity
    P = build_problem(orders, Params(n_vehicles=n_vehicles, capacity=capacity, traffic=traffic))
    sol = solve_classical(P)

    # 16-qubit windows are ~100x heavier than 9-qubit windows, so use a lighter optimiser budget
    if k == 4:
        opts = dict(maxiter=80, restarts=1, max_windows=4)
    else:
        opts = dict(maxiter=150, restarts=3, max_windows=24)
    info = {}
    t0 = time.time()
    routes, logs = hybrid_solve(P, sol["nearest_neighbour"], k=k, reps=reps, seed=7, verbose=False,
                                time_limit=TIME_LIMIT_S, info=info, **opts)
    qaoa_time = time.time() - t0

    methods = {}
    for key, label, r in [("fcfs", "FCFS (baseline)", sol["fcfs"]),
                          ("nn", "Nearest neighbour", sol["nearest_neighbour"]),
                          ("two_opt", "Classical 2-opt", sol["two_opt"]),
                          ("hybrid", "Hybrid: NN + QAOA", routes)]:
        methods[key] = {"label": label, "routes": r, "metrics": evaluate_solution(P, r)}

    summary = {"day": orders.attrs["ds"], "orders": n, "vehicles": n_vehicles, "capacity": capacity,
               "k": k, "reps": reps, "seed": seed, "windows": len(logs),
               "truncated": bool(info.get("truncated")), "qaoa_time": qaoa_time}
    windows = []
    if logs:
        try:
            diff, depth = logs[0]["engine"].verify_with_qiskit()
            qiskit = {"max_diff": diff, "depth": depth}
        except Exception as e:                       # qiskit missing or API difference
            qiskit = {"error": str(e)[:120]}
        for w in logs:
            w = dict(w)
            w.pop("engine")
            w["label"] = f"Vehicle {w['vehicle']} - window @{w['start']}"
            windows.append(w)
        nq = len(logs[0]["tours"])
        summary.update({
            "qubits": max(l["qubits"] for l in logs),
            "improved_windows": sum(l["improved"] for l in logs),
            "top1_optimal": sum(l["qaoa_top_is_optimal"] for l in logs),
            "avg_valid_after": sum(l["valid_prob_layers"][-1] for l in logs) / len(logs),
            "valid_uniform": logs[0]["valid_prob_uniform"],
            "avg_rank": sum(l["qaoa_top_rank"] for l in logs) / len(logs),
            "random_rank": (nq - 1) / 2,
            "qiskit": qiskit,
        })
    points = [{"id": P.ids[i], "lat": P.lat[i], "lng": P.lng[i],
               "ready": P.ready[i], "due": P.due[i]} for i in range(len(P.ids))]
    return clean({"summary": summary, "points": points, "methods": methods, "windows": windows})


if __name__ == "__main__":
    import json
    r = run_optimization()
    print(json.dumps(r["summary"], indent=2))
    for k, m in r["methods"].items():
        print(f"{m['label']:<22}", m["metrics"])