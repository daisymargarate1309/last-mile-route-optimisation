"""Step 6: constructive / destructive interference inside QAOA.

Last mixer:  amplitude(z) = sum over ALL basis states x of  M(z,x) * phi(x)
    phi(x)   = state just before the last mixer (carries the cost phase e^{-i g C(x)})
    M(z,x)   = cos(b)^(n-d) * (-i sin b)^d ,  d = Hamming distance(z, x)
Each term is a 'path' that reaches z. Paths are complex numbers (arrows):
    arrows point the same way  -> they ADD   -> constructive interference -> probability goes UP
    arrows point opposite ways -> they CANCEL -> destructive interference -> probability goes DOWN
Interference factor = |sum of paths|^2 / sum(|path|^2)
    > 1  constructive   (more than the 'no interference' value)
    < 1  destructive
"""
import numpy as np

BINS = 12


def path_analysis(engine, phi, beta, z_index):
    x = np.arange(engine.N)
    d = engine.popcount[x ^ z_index]
    M = np.cos(beta) ** (engine.n - d) * (-1j * np.sin(beta)) ** d
    c = M * phi                                           # every path that ends in z
    amp = c.sum()
    no_interf = float(np.sum(np.abs(c) ** 2))             # what we'd get if paths just added as probabilities
    prob = float(abs(amp) ** 2)
    hist, _ = np.histogram(np.angle(c), bins=BINS, range=(-np.pi, np.pi), weights=np.abs(c))
    return {
        "probability": prob,
        "no_interference": no_interf,
        "factor": prob / no_interf if no_interf > 0 else 0.0,
        "coherence": float(abs(amp) / np.abs(c).sum()),    # 1 = all arrows aligned, 0 = fully cancelled
        "phase": float(np.angle(amp)),
        "phase_hist": (hist / max(hist.sum(), 1e-12)).round(4).tolist(),   # arrows by direction
    }


def label(factor):
    return "constructive" if factor > 1.05 else ("destructive" if factor < 0.95 else "neutral")