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

from test.common import QiskitAquaChemistryTestCase
from qiskit_chemistry import QiskitChemistryError
from qiskit_chemistry.drivers import GaussianDriver
from test.test_driver import TestDriver


class TestDriverGaussian(QiskitAquaChemistryTestCase, TestDriver):
    """Gaussian Driver tests."""

    def setUp(self):
        try:
            driver = GaussianDriver([
                    '# rhf/sto-3g scf(conventional) geom=nocrowd',
                    '',
                    'h2 molecule',
                    '',
                    '0 1',
                    'H   0.0  0.0    0.0',
                    'H   0.0  0.0    0.735',
                    ''
                    ])
        except QiskitChemistryError:
            self.skipTest('GAUSSIAN driver does not appear to be installed')
        self.qmolecule = driver.run()


if __name__ == '__main__':
    unittest.main()
