#!/usr/bin/env python 

import unittest

from rdkit.Chem.rdchem import Atom
from rdkit.Chem import AllChem

from chem_utils.chem_handler import AtomHandler


class AtomHandlerTest(unittest.TestCase):

    def test_dropped_atom1(self):
        atom = AllChem.MolFromSmiles("CCC").GetAtomWithIdx(0)
        AtomHandler.set_atom_is_dropped(atom)
        self.assertTrue(AtomHandler.is_atom_dropped(atom))

    def test_dropped_atom2(self):
        atom = AllChem.MolFromSmiles("CCC").GetAtomWithIdx(0)
        self.assertFalse(AtomHandler.is_atom_dropped(atom))


if __name__ == "__main__":
    unittest.main()
