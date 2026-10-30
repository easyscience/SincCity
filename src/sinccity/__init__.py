"""Generic Scattering Calculator: I(q) of a real-space structure.

A notebook equivalent of the nuclear, 1D part of SasView's Generic Scattering
Calculator: load a PDB, SLD or VTK file, compute the orientationally averaged
I(q) by the exact Debye formula, and show the points and the curve. The
result converts to a sasdata ``Data1D``, which is how it reaches a fit (for
example ``SANSFitter.set_data`` in sans-fitter).

Example:
    >>> from sinccity import GenericScatteringCalculator
    >>> gsc = GenericScatteringCalculator()
    >>> gsc.load_structure('particle.sld')
    >>> gsc.plot_points()
    >>> gsc.set_params(sld_solvent=6.34)     # 10^-6 1/Ang^2, as in sasmodels
    >>> profile = gsc.calculate()
    >>> gsc.plot_profile()
    >>> data = profile.to_data1d()          # a Data1D for SANSFitter.set_data
"""

__version__ = '0.1.0'

from ._console import set_verbosity
from .calculator import GenericScatteringCalculator, ScatteringProfile
from .debye import CalculationInterrupted
from .model import RgResult, Structure, StructureError
from .readers import load, read_pdb, read_sld, read_vtk

__all__ = [
    'CalculationInterrupted',
    'GenericScatteringCalculator',
    'RgResult',
    'ScatteringProfile',
    'Structure',
    'StructureError',
    'load',
    'read_pdb',
    'read_sld',
    'read_vtk',
    'set_verbosity',
]
