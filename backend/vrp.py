"""Step 2: the problem definition (capacity, time windows, traffic, cost) and route evaluation."""
from dataclasses import dataclass
import numpy as np

TRAFFIC = {"free": 1.0, "normal": 1.25, "peak": 1.6}   # travel-time multiplier


@dataclass
class Params:
    n_vehicles: int = 3
    capacity: int = 8            # max orders per vehicle (each order has demand 1)
    service_min: float = 5.0     # minutes spent at each stop
    speed_kmph: float = 25.0     # average city speed
    detour: float = 1.3          # road distance = straight-line * detour
    traffic: str = "normal"      # free / normal / peak
    fixed_cost: float = 50.0     # cost of using one vehicle
    cost_per_km: float = 8.0     # fuel + running cost
    late_penalty: float = 2.0    # cost per minute of lateness


@dataclass
class Problem:
    ids: list
    lat: np.ndarray
    lng: np.ndarray
    demand: np.ndarray
    ready: np.ndarray
    due: np.ndarray
    dist: np.ndarray    # road km
    time: np.ndarray    # travel minutes (with traffic)
    params: Params

    @property
    def n(self):        # number of customers (node 0 is the depot)
        return len(self.ids) - 1


def haversine_km(lat, lng):
    la, lo = np.radians(lat), np.radians(lng)
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    a = np.sin(dla / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def build_problem(orders, params=None):
    p = params or Params()
    lat = np.r_[orders["lat"].mean(), orders["lat"].values]      # depot = centre of the orders
    lng = np.r_[orders["lng"].mean(), orders["lng"].values]
    dist = haversine_km(lat, lng) * p.detour
    time = dist / p.speed_kmph * 60.0 * TRAFFIC[p.traffic]
    return Problem(
        ids=["DEPOT"] + [str(i) for i in orders["order_id"]],
        lat=lat, lng=lng,
        demand=np.r_[0, np.ones(len(orders))],
        ready=np.r_[0.0, orders["ready"].values.astype(float)],
        due=np.r_[1e9, orders["due"].values.astype(float)],
        dist=dist, time=time, params=p,
    )


def evaluate_route(P, route):
    """Simulate one vehicle: depot -> route -> depot."""
    if not route:
        return dict(distance=0.0, duration=0.0, load=0, late_count=0, late_min=0.0, wait=0.0)
    t = max(0.0, P.ready[route[0]] - P.time[0, route[0]])    # leave depot just in time
    depart, prev = t, 0
    dist = late = wait = 0.0
    late_count = 0
    for c in route:
        t += P.time[prev, c]
        dist += P.dist[prev, c]
        if t < P.ready[c]:
            wait += P.ready[c] - t
            t = P.ready[c]
        elif t > P.due[c]:
            late += t - P.due[c]
            late_count += 1
        t += P.params.service_min
        prev = c
    t += P.time[prev, 0]
    dist += P.dist[prev, 0]
    return dict(distance=dist, duration=t - depart, load=int(P.demand[route].sum()),
                late_count=late_count, late_min=late, wait=wait)


def route_cost(P, route):
    """Objective used by the optimisers (lower is better)."""
    if not route:
        return 0.0
    m = evaluate_route(P, route)
    over = max(0, m["load"] - P.params.capacity)
    return (P.params.fixed_cost + P.params.cost_per_km * m["distance"]
            + P.params.late_penalty * m["late_min"] + 1000.0 * over)


def evaluate_solution(P, routes):
    ms = [evaluate_route(P, r) for r in routes]
    used = sum(1 for r in routes if r)
    return {
        "vehicles_used": used,
        "distance": round(sum(m["distance"] for m in ms), 2),
        "duration": round(sum(m["duration"] for m in ms), 1),
        "late_count": sum(m["late_count"] for m in ms),
        "late_min": round(sum(m["late_min"] for m in ms), 1),
        "cost": round(sum(route_cost(P, r) for r in routes), 1),
        "feasible": all(m["load"] <= P.params.capacity for m in ms)
                    and sum(m["late_count"] for m in ms) == 0,
    }