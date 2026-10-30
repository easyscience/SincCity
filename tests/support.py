"""Helpers shared by the sinccity tests."""

import json
import os
import warnings

import numpy as np

from sinccity import GenericScatteringCalculator, Structure

DATA = os.path.join(os.path.dirname(__file__), 'data')
C60 = os.path.join(DATA, 'c60.pdb')
CUBE_VTK = os.path.join(DATA, 'five_tetrahedra_cube.vtk')
Q = np.r_[0.0, np.geomspace(1e-3, 3.0, 25)]


def brute(q, xyz, w):
    """The Debye sum written out in full, as a reference for the kernels."""
    r = np.sqrt(((xyz[:, None] - xyz[None]) ** 2).sum(-1))
    return np.array([(w[:, None] * w[None] * np.sinc(qi * r / np.pi)).sum() for qi in q])


def close(a, b, w):
    """Kernel agreement on the scale of the terms, not of D (which can be tiny)."""
    return np.max(np.abs(a - b)) <= 1e-12 * np.abs(w).sum() ** 2


def sphere_pixels(radius=30.0, step=3.0, sld=1.0, sld_solvent=6.34):
    """A sphere of pixels in a box of solvent pixels (SLD in 10^-6 1/Ang^2)."""
    n = int(np.ceil(radius / step)) + 1
    axis = np.arange(-n, n + 1) * step
    xyz = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    inside = (xyz**2).sum(1) <= radius**2
    values = np.where(inside, sld, sld_solvent) * 1e-6
    return Structure.from_arrays(xyz, values, np.full(len(xyz), step**3))


def gro_text(atoms, box='   5.00000   6.00000   7.00000', ndec=3):
    """A .gro file; atoms are (resid, resname, name, x, y, z) in nm."""
    width = ndec + 5
    lines = ['written by a test', f'{len(atoms):5d}']
    for number, (resid, resname, name, *xyz) in enumerate(atoms, start=1):
        position = ''.join(f'{v:{width}.{ndec}f}' for v in xyz)
        lines.append(f'{resid % 100000:>5d}{resname:<5}{name:>5}{number % 100000:>5d}{position}')
    return '\n'.join([*lines, box]) + '\n'


def run_task(task: str) -> list[list[float]]:
    """What a worker process runs: a JSON settings record and series, as shapespyer sends them."""
    import sinccity

    sinccity.set_verbosity('quiet')
    settings, series = json.loads(task)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        calculator = GenericScatteringCalculator.from_settings(settings)
        return [p.intensity.tolist() for p in calculator.calculate_series(series)]


def quiet_calculator(structure=None, qmax=None):
    """A calculator whose load-time warnings (reader notes) are not reported."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gsc = GenericScatteringCalculator(structure)
    if qmax is not None:
        gsc.set_q_grid(qmax=qmax)
    return gsc
