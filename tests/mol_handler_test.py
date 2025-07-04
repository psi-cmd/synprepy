#!/usr/bin/env python 

import unittest

from rdkit.Chem import AllChem

from chem_utils.chem_handler import MolHandler


class MolHandlerTest(unittest.TestCase):

    def test_query_atom_with_map_num(self):
        mol = AllChem.MolFromSmiles("[CH3:1][CH2:2][O:1]C")
        atoms = MolHandler.query_atoms_with_map_num(mol, 1)
        self.assertEqual(len(list(atoms)), 2)


if __name__ == "__main__":
    unittest.main()
