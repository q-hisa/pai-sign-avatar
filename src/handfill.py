"""Method D: hand-shape-prior gap filling.

Interpolate in a canonical frame (shape = Procrustes-aligned landmarks, orientation = slerp of rotations)
and project the interpolated shape onto a PCA subspace learned from detected hands (plausible hand shapes).
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def kabsch(src, dst):
    U, _, Vt = np.linalg.svd(src.T @ dst)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T  # dst ~= src @ R.T


class ShapePrior:
    def __init__(self, library, k=8):
        """library: (N,21,3) detected hands (any orientation)."""
        lib = [h - h.mean(0) for h in library]
        self.ref = lib[0]
        for _ in range(3):  # iterate reference (generalized Procrustes)
            aligned = [x @ kabsch(x, self.ref).T for x in lib]
            self.ref = np.mean(aligned, 0)
        aligned = np.array([x @ kabsch(x, self.ref).T for x in lib]).reshape(len(lib), -1)
        self.mean = aligned.mean(0)
        _, _, Vt = np.linalg.svd(aligned - self.mean, full_matrices=False)
        self.V = Vt[:k]

    def canon(self, h):
        c = h.mean(0)
        x = h - c
        R = kabsch(x, self.ref)  # canon = x @ R.T
        return x @ R.T, R, c

    def project(self, canon):
        z = (canon.reshape(-1) - self.mean) @ self.V.T
        return (self.mean + z @ self.V).reshape(21, 3)

    def fill(self, a, b, L, project=True):
        """a, b: detected hands before/after the gap. returns (L,21,3)."""
        ca, Ra, pa = self.canon(a)
        cb, Rb, pb = self.canon(b)
        sl = Slerp([0, 1], Rotation.from_matrix([Ra.T, Rb.T]))  # Ra.T maps canon -> original orientation
        out = []
        for i in range(1, L + 1):
            w = i / (L + 1)
            c = (1 - w) * ca + w * cb
            if project:
                c = self.project(c)
            R = sl([w]).as_matrix()[0]
            out.append(c @ R.T + ((1 - w) * pa + w * pb))
        return np.array(out)
