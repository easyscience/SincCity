"""Element data and guesses, the residue steps, and structures derived from others."""

import numpy as np
import pytest
from support import C60, quiet_calculator

from sinccity import GenericScatteringCalculator, Structure, StructureError, read_pdb
from sinccity.elements import constants, element_from_mass, guess_stem
from sinccity.molecules import assign_elements

# ============================================================================
# Elements
# ============================================================================


class TestElements:
    def test_masses_give_elements_and_deuterium(self):
        symbols = element_from_mass([1.008, 2.014, 12.011, 14.007, 35.453, 1.00794])
        assert symbols.tolist() == ['H', 'D', 'C', 'N', 'Cl', 'H']

    @pytest.mark.parametrize('mass', [3.024, 14.027])  # mass-repartitioned H, united-atom CH2
    def test_masses_that_are_no_element(self, mass):
        with pytest.raises(StructureError, match="no element's"):
            element_from_mass([12.011, mass])

    def test_stem_rule(self):
        assert guess_stem('1HB', single_atom=False) == 'H'
        assert guess_stem('C21', single_atom=False) == 'C'
        assert guess_stem('CA', single_atom=False) == 'C'  # alpha carbon in a protein
        assert guess_stem('CA', single_atom=True) == 'Ca'  # a calcium ion
        assert guess_stem('NA', single_atom=True) == 'Na'
        assert guess_stem('OW', single_atom=True) == 'O'  # not an element: first letter

    def test_unknown_symbol(self):
        with pytest.raises(StructureError, match="'Xx'"):
            constants('Xx')


# ============================================================================
# Element assignment
# ============================================================================


def atoms(*residues):
    """resids, resnames and names for residues given as (resid, resname, names)."""
    rows = [(resid, resname, name) for resid, resname, names in residues for name in names]
    return tuple(np.array(column) for column in zip(*rows))


class TestAssignElements:
    def test_sources_in_order_and_by_position(self):
        resids, resnames, names = atoms(
            (1, 'CTA', ['NX', 'CX', 'HX']),  # names unlike the topology's
            (2, 'CTA', ['NX', 'CX', 'HX']),
            (3, 'BR', ['BR']),
            (4, 'ION', ['K']),
            (5, 'LIG', ['C1', 'O1']),
        )
        file_elements = np.array(['', '', '', '', '', '', '', '', 'C', 'O'])
        symbols, sources, note = assign_elements(
            resids,
            resnames,
            names,
            masses={'CTA': [14.007, 12.011, 2.014], 'ION': [39.098]},
            elements={'CTA': ['C', 'C', 'C'], 'ION': ['Na']},
            file_elements=file_elements,
        )
        assert symbols.tolist() == ['N', 'C', 'D', 'N', 'C', 'D', 'Br', 'K', 'C', 'O']
        assert sources == {'BR': 'names', 'CTA': 'masses', 'ION': 'masses', 'LIG': 'pdb'}
        assert note.startswith('Elements of BR guessed') and 'BR -> Br in BR' in note

    def test_partial_element_column_falls_back_to_the_guess(self):
        resids, resnames, names = atoms((1, 'LIG', ['C1', 'O1']))
        symbols, sources, _ = assign_elements(
            resids, resnames, names, {}, {}, np.array(['C', '']), np.array(['C', 'O'])
        )
        assert symbols.tolist() == ['C', 'O'] and sources == {'LIG': 'names'}

    def test_refusals(self):
        resids, resnames, names = atoms((1, 'CTA', ['N1', 'C1']))
        with pytest.raises(StructureError, match='does not have the 3 atoms'):
            assign_elements(resids, resnames, names, {'CTA': [14.0, 12.0, 1.0]}, {})
        with pytest.raises(StructureError, match='CTA: The mass 3.024'):
            assign_elements(resids, resnames, names, {'CTA': [14.007, 3.024]}, {})
        resids, resnames, names = atoms((1, 'X', ['1', 'Q1']))
        with pytest.raises(StructureError, match='atom names of X .*elements='):
            assign_elements(resids, resnames, names, {}, {})


# ============================================================================
# Derived structures
# ============================================================================


class TestDerivedStructures:
    def test_from_atoms(self):
        s = Structure.from_atoms(
            [[0.0, 0, 0], [1.5, 0, 0]], ['c', 'CL'], resids=[1, 2], resnames=['A', 'B']
        )
        assert s.symbols.tolist() == ['C', 'Cl'] and s.format == 'arrays'
        assert s.sld[0] == pytest.approx(constants('C')[0])
        assert s.volume[1] == pytest.approx(constants('Cl')[1])
        assert s.element_sources == {'A': 'given', 'B': 'given'}
        assert s.reader_options is None
        settings = quiet_calculator(s).get_settings()
        assert settings['source'] == 'in-memory' and 'derived_from' not in settings

    def test_with_symbols_is_a_derived_copy(self):
        s = read_pdb(C60)
        symbols = s.symbols.copy()
        symbols[0] = 'N'
        fixed = s.with_symbols(symbols)
        assert fixed.symbols[0] == 'N' and fixed.sld[0] == pytest.approx(constants('N')[0])
        assert fixed.element_sources == {'???': 'elements'} and s.element_sources == {
            '???': 'names'
        }
        assert fixed.source == s.source and fixed.reader_options is None
        assert s.symbols[0] == 'C'  # the original is unchanged
        with pytest.raises(StructureError, match='59 elements given for 60'):
            s.with_symbols(symbols[1:])

        settings = quiet_calculator(fixed).get_settings()
        assert settings['source'] == 'in-memory' and settings['derived_from'] == s.source
        with pytest.raises(ValueError, match='set_structure'):
            GenericScatteringCalculator.from_settings(settings)

    def test_summary_lists_residues_and_sources(self):
        s = read_pdb(C60)
        info = s.summary()
        assert info['n_residues'] == 1 and info['residue_atoms'] == {'???': 60}
        assert info['element_sources'] == {'???': 'names'}
        assert ('Elements from', '??? names') in s._lines()
