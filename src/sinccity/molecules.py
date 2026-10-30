"""Residue-level steps shared by the PDB and GRO readers and by :class:`Structure`."""

import numpy as np


def residue_runs(resids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First index and atom count of each residue: a run of atoms with one residue number."""
    resids = np.asarray(resids)
    starts = np.flatnonzero(np.r_[True, resids[1:] != resids[:-1]])
    return starts, np.diff(np.r_[starts, resids.size])
