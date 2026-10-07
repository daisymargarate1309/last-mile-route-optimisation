"""Run:  python -m backend.run_classical   (from the project root)"""
from backend.lade_data import select_orders, list_days
from backend.vrp import Params, build_problem, evaluate_solution
from backend.classical_solvers import solve_classical

if __name__ == "__main__":
    mode = "pickup"                      # "pickup" has real time windows
    print("Busiest days:\n", list_days(mode, 5), "\n")

    orders = select_orders(mode=mode, n=20, seed=42)
    print(f"Day {orders.attrs['ds']}: {len(orders)} orders selected")
    print(orders.head(), "\n")

    P = build_problem(orders, Params(n_vehicles=4, capacity=8, traffic="normal"))
    sol = solve_classical(P)

    print(f"{'method':<20}{'veh':>4}{'km':>8}{'min':>8}{'late':>6}{'late-min':>10}{'cost':>9}  feasible")
    for name in ["fcfs", "nearest_neighbour", "two_opt"]:
        m = evaluate_solution(P, sol[name])
        print(f"{name:<20}{m['vehicles_used']:>4}{m['distance']:>8}{m['duration']:>8}"
              f"{m['late_count']:>6}{m['late_min']:>10}{m['cost']:>9}  {m['feasible']}")
    print("\nRoutes (2-opt):")
    for k, r in enumerate(sol["two_opt"], 1):
        print(f"  vehicle {k}: DEPOT -> " + " -> ".join(P.ids[i] for i in r) + " -> DEPOT")