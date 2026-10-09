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
def optimize(mode: str = "pickup", n_orders: str = "20", n_vehicles: str = "4", capacity: str = "8",
             traffic: str = "normal", k: str = "3", reps: str = "3", seed: str = "42"):
    # values are validated / clamped inside run_optimization, so any input still gives an answer
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(429, "Another run is in progress. Please wait.")
    try:
        return run_optimization(mode=mode, n_orders=n_orders, n_vehicles=n_vehicles, capacity=capacity,
                                traffic=traffic, k=k, reps=reps, seed=seed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        _run_lock.release()


FRONTEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
os.makedirs(FRONTEND, exist_ok=True)
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")