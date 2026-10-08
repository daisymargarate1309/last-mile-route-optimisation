"""Step 8: one function that runs the whole pipeline and returns JSON-friendly data for the website."""
import time
import numpy as np
from backend.lade_data import select_orders
from backend.vrp import Params, build_problem, evaluate_solution
from backend.classical_solvers import solve_classical
from backend.hybrid import hybrid_solve


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


def run_optimization(mode="pickup", n_orders=20, n_vehicles=4, capacity=8, traffic="normal",
                     k=3, reps=3, seed=42, radius_km=6.0):
    orders = select_orders(mode=mode, n=n_orders, seed=seed, radius_km=radius_km)
    if len(orders) < 4:
        raise ValueError("Too few orders found for this day/area. Try another seed.")
    P = build_problem(orders, Params(n_vehicles=n_vehicles, capacity=capacity, traffic=traffic))
    sol = solve_classical(P)

    t0 = time.time()
    routes, logs = hybrid_solve(P, sol["nearest_neighbour"], k=k, reps=reps, seed=7, verbose=False)
    qaoa_time = time.time() - t0
    if not logs:
        raise ValueError("Routes are too short for QAOA windows. Use fewer vehicles.")

    methods = {}
    for key, label, r in [("fcfs", "FCFS (baseline)", sol["fcfs"]),
                          ("nn", "Nearest neighbour", sol["nearest_neighbour"]),
                          ("two_opt", "Classical 2-opt", sol["two_opt"]),
                          ("hybrid", "Hybrid: NN + QAOA", routes)]:
        methods[key] = {"label": label, "routes": r, "metrics": evaluate_solution(P, r)}

    try:
        diff, depth = logs[0]["engine"].verify_with_qiskit()
        qiskit = {"max_diff": diff, "depth": depth}
    except Exception as e:                       # qiskit missing or API difference
        qiskit = {"error": str(e)[:120]}

    windows = []
    for w in logs:
        w = dict(w)
        w.pop("engine")
        w["label"] = f"Vehicle {w['vehicle']} - window @{w['start']}"
        windows.append(w)

    nq = len(logs[0]["tours"])
    summary = {
        "day": orders.attrs["ds"], "orders": len(orders), "windows": len(logs),
        "qubits": logs[0]["qubits"], "reps": reps, "k": k,
        "improved_windows": sum(l["improved"] for l in logs),
        "top1_optimal": sum(l["qaoa_top_is_optimal"] for l in logs),
        "avg_valid_after": sum(l["valid_prob_layers"][-1] for l in logs) / len(logs),
        "valid_uniform": logs[0]["valid_prob_uniform"],
        "avg_rank": sum(l["qaoa_top_rank"] for l in logs) / len(logs),
        "random_rank": (nq - 1) / 2,
        "qaoa_time": qaoa_time, "qiskit": qiskit,
    }
    points = [{"id": P.ids[i], "lat": P.lat[i], "lng": P.lng[i],
               "ready": P.ready[i], "due": P.due[i]} for i in range(len(P.ids))]
    return clean({"summary": summary, "points": points, "methods": methods, "windows": windows})


if __name__ == "__main__":
    import json
    r = run_optimization()
    print(json.dumps(r["summary"], indent=2))
    for k, m in r["methods"].items():
        print(f"{m['label']:<22}", m["metrics"])