"""Step 9: FastAPI backend. Serves the optimisation API and the frontend page."""
import os
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from backend.lade_data import load_raw
from backend.service import run_optimization

_run_lock = threading.Lock()          # one optimisation at a time


@asynccontextmanager
async def lifespan(app):
    # read the big CSV in the background so the first click is fast
    threading.Thread(target=lambda: load_raw("pickup"), daemon=True).start()
    yield


app = FastAPI(title="Quantum-Enhanced Last-Mile Route Optimisation", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/optimize")
def optimize(mode: str = "pickup", n_orders: int = 20, n_vehicles: int = 4, capacity: int = 8,
             traffic: str = "normal", k: int = 3, reps: int = 3, seed: int = 42):
    if mode not in ("pickup", "delivery"):
        raise HTTPException(400, "mode must be pickup or delivery")
    if traffic not in ("free", "normal", "peak"):
        raise HTTPException(400, "traffic must be free, normal or peak")
    if not 8 <= n_orders <= 40:
        raise HTTPException(400, "n_orders must be between 8 and 40")
    if k not in (3, 4):
        raise HTTPException(400, "k (stops per quantum window) must be 3 or 4")
    if not 1 <= reps <= 4:
        raise HTTPException(400, "reps must be between 1 and 4")
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(429, "Another optimisation is running. Please wait.")
    try:
        return run_optimization(mode=mode, n_orders=n_orders, n_vehicles=n_vehicles,
                                capacity=capacity, traffic=traffic, k=k, reps=reps, seed=seed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        _run_lock.release()


FRONTEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
os.makedirs(FRONTEND, exist_ok=True)
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")