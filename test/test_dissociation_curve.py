# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import unittest
import numpy as np
from test.common import QiskitAquaChemistryTestCase
from qiskit.chemistry import DissociationCurve


class TestDissociationCurve(QiskitAquaChemistryTestCase):
    """DissociationCurve tests."""

    def setUp(self):
        self.h2_dc = DissociationCurve("H2", 'H 0.0 0.0 0.0; H 0.0 0.0 {0}', list(np.arange(.6, .8, .1)),
                                       readonly_mode=True)
        self.reference_energies = []

    # @parameterized.expand([
    #     ['COBYLA_M', 'COBYLA', qiskit.Aer.get_backend('statevector_simulator'), 'matrix', 1],
    #     ['COBYLA_P', 'COBYLA', qiskit.Aer.get_backend('statevector_simulator'), 'paulis', 1],
    #     ['SPSA_P', 'SPSA', 'qasm_simulator', 'paulis', 1024],
    #     ['SPSA_GP', 'SPSA', 'qasm_simulator', 'grouped_paulis', 1024]
    # ])
    def test_end2end_dc(self):
        #, name, optimizer, backend, mode, shots):
        self.h2_dc.run()
        np.testing.assert_array_almost_equal(self.h2_dc._vqe_results['results'][.6], self.reference_energies, decimal=6)


if __name__ == '__main__':
    unittest.main()
