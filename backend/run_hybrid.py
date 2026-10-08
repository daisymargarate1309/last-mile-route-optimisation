"""Run:  python -m backend.run_hybrid        (from the project root)"""
import time
from backend.lade_data import select_orders
from backend.vrp import Params, build_problem, evaluate_solution
from backend.classical_solvers import solve_classical
from backend.hybrid import hybrid_solve

K = 3          # stops per QAOA window  (3 -> 9 qubits, 4 -> 16 qubits)
REPS = 3       # QAOA layers p

if __name__ == "__main__":
    print("Loading data ...", flush=True)
    orders = select_orders(mode="pickup", n=20, seed=42)
    P = build_problem(orders, Params(n_vehicles=4, capacity=8, traffic="normal"))
    sol = solve_classical(P)

    print("Running QAOA windows ...")
    t0 = time.time()
    routes, logs = hybrid_solve(P, sol["nearest_neighbour"], k=K, reps=REPS)
    print(f"QAOA time: {time.time()-t0:.1f}s\n")

    print(f"{'method':<28}{'veh':>4}{'km':>8}{'min':>8}{'late':>6}{'cost':>9}")
    rows = [("FCFS (baseline)", sol["fcfs"]), ("Nearest neighbour", sol["nearest_neighbour"]),
            ("Classical 2-opt", sol["two_opt"]), ("HYBRID NN + QAOA windows", routes)]
    for name, r in rows:
        m = evaluate_solution(P, r)
        print(f"{name:<28}{m['vehicles_used']:>4}{m['distance']:>8}{m['duration']:>8}"
              f"{m['late_count']:>6}{m['cost']:>9}")

    ok = sum(l["qaoa_top_is_optimal"] for l in logs)
    imp = sum(l["improved"] for l in logs)
    print(f"\nWindows: {len(logs)} | QAOA's #1 tour was the true best: {ok}/{len(logs)} | "
          f"windows that improved the route: {imp}")
    avg_valid = sum(l["valid_prob_layers"][-1] for l in logs) / len(logs)
    avg_rank = sum(l["qaoa_top_rank"] for l in logs) / len(logs)
    print(f"Average P(valid tour): {logs[0]['valid_prob_uniform']:.3f} (uniform) -> {avg_valid:.3f} (after QAOA)")
    print(f"Average rank of QAOA's #1 tour: {avg_rank:.2f}  (0 = best, random guessing = "
          f"{(len(logs[0]['tours'])-1)/2:.1f})")
    # ---------------- interference report (first window) ----------------
    w = logs[0]
    print(f"\n=== INTERFERENCE in vehicle {w['vehicle']} window@{w['start']} ({w['qubits']} qubits) ===")
    print("P(any valid tour) after each layer:",
          " -> ".join(f"{p:.3f}" for p in w["valid_prob_layers"]), "(layer 0 = uniform superposition)")
    print(f"{'tour':<26}{'exact cost':>11}{'P uniform':>11}{'P after QAOA':>14}{'factor':>8}  type")
    for t in sorted(w["tours"], key=lambda t: t["exact_cost"]):
        print(f"{' > '.join(t['tour']):<26}{t['exact_cost']:>11}{t['p_uniform']:>11.4f}"
              f"{t['p_final']:>14.4f}{t['factor']:>8}  {t['type']}")
    pv = w["valid_prob_layers"][-1]
    print(f"\nValid tours : {w['valid_prob_uniform']:.3f} -> {pv:.3f}  (constructive: probability piles up here)")
    print(f"Invalid bits: {1-w['valid_prob_uniform']:.3f} -> {1-pv:.3f}  (destructive: their amplitudes cancel)")
    print("factor = |sum of paths|^2 / sum |path|^2  (>1 constructive, <1 destructive)")

    try:
        diff, depth = w["engine"].verify_with_qiskit()
        print(f"\nQiskit check: max |numpy - Qiskit Statevector| = {diff:.2e}, circuit depth = {depth}")
    except ImportError:
        print("\n(install qiskit to run the Qiskit verification: pip install qiskit)")