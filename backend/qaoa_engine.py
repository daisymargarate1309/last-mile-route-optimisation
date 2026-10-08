"""Step 5: QAOA. Fast exact state-vector simulation in numpy (+ a Qiskit circuit for verification).
Qubit q  <->  bit q of the basis-state index (same convention as Qiskit)."""
import numpy as np
from scipy.optimize import minimize


def qubo_to_ising(Q):
    """x = (1 - Z)/2  ->  single-Z weights h and ZZ weights J."""
    n = Q.shape[0]
    h, J = {}, {}
    for a in range(n):
        if Q[a, a] != 0:
            h[a] = h.get(a, 0.0) - Q[a, a] / 2
        for b in range(a + 1, n):
            w = Q[a, b]
            if w != 0:
                J[(a, b)] = J.get((a, b), 0.0) + w / 4
                h[a] = h.get(a, 0.0) - w / 4
                h[b] = h.get(b, 0.0) - w / 4
    return h, J


class QAOA:
    def __init__(self, Q, reps=2):
        self.Q, self.reps = Q, reps
        self.gamma_max = 1.0
        self.n = Q.shape[0]
        self.N = 2 ** self.n
        x = np.arange(self.N)
        bits = ((x[:, None] >> np.arange(self.n)) & 1).astype(float)
        self.costs = np.sum((bits @ Q) * bits, axis=1)        # QUBO cost of every bitstring
        self.popcount = bits.sum(axis=1).astype(int)

    # ---- one QAOA layer = cost phase + mixer ----------------------------------------
    def _mixer(self, psi, beta):
        c, s = np.cos(beta), -1j * np.sin(beta)
        n = self.n
        for q in range(n):
            v = psi.reshape(2 ** (n - 1 - q), 2, 2 ** q)
            a, b = v[:, 0, :].copy(), v[:, 1, :].copy()
            v[:, 0, :] = c * a + s * b
            v[:, 1, :] = s * a + c * b
        return psi

    def run(self, gammas, betas):
        """Returns final state, the state just BEFORE the last mixer, and the state after every layer."""
        psi = np.full(self.N, 1 / np.sqrt(self.N), dtype=complex)    # |+>^n : uniform superposition
        layers, phi = [psi.copy()], None
        for g, b in zip(gammas, betas):
            psi = psi * np.exp(-1j * g * self.costs)                 # phase kick  e^{-i g C(x)}
            phi = psi.copy()
            psi = self._mixer(psi, b)                                # interference happens here
            layers.append(psi.copy())
        return psi, phi, layers

    def _start(self, rng, r_i):
        """Restart 0: annealing-like ramp (gamma grows, beta shrinks). Others: ramp + noise."""
        p = self.reps
        ramp = (np.arange(p) + 0.5) / p
        g = self.gamma_max * ramp
        b = (np.pi / 4) * (1 - ramp)
        if r_i > 0:
            g = g * rng.uniform(0.5, 1.5, p)
            b = b * rng.uniform(0.5, 1.5, p)
        return np.concatenate([g, b])

    def optimize(self, maxiter=120, restarts=2, seed=7):
        rng = np.random.default_rng(seed)
        p, best = self.reps, None
        for r_i in range(restarts):
            hist = []

            def f(x):
                psi, _, _ = self.run(x[:p], x[p:])
                e = float(np.dot(np.abs(psi) ** 2, self.costs))
                hist.append(e)
                return e

            x0 = self._start(rng, r_i)
            r = minimize(f, x0, method="COBYLA", options={"maxiter": maxiter})
            if best is None or r.fun < best[0].fun:
                best = (r, hist)
        r, hist = best
        self.gammas, self.betas = r.x[:p], r.x[p:]
        return {"gammas": self.gammas, "betas": self.betas, "history": hist, "energy": float(r.fun)}

    # ---- Qiskit circuit (same maths) ---------------------------------------------------
    def qiskit_circuit(self):
        from qiskit import QuantumCircuit
        h, J = qubo_to_ising(self.Q)
        qc = QuantumCircuit(self.n)
        qc.h(range(self.n))
        for g, b in zip(self.gammas, self.betas):
            for (i, j), w in J.items():
                qc.rzz(2 * g * w, i, j)
            for i, w in h.items():
                qc.rz(2 * g * w, i)
            for q in range(self.n):
                qc.rx(2 * b, q)
        return qc

    def verify_with_qiskit(self):
        """Max difference between our numpy probabilities and Qiskit's Statevector (should be ~1e-12)."""
        from qiskit.quantum_info import Statevector
        qc = self.qiskit_circuit()
        mine = np.abs(self.run(self.gammas, self.betas)[0]) ** 2
        theirs = Statevector(qc).probabilities()
        return float(np.max(np.abs(mine - theirs))), qc.depth()