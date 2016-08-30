#!/usr/bin/env python
from __future__ import absolute_import, division

#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

"""
Sogo Mineo writes:

'''
If I read Wcs from, e.g., the following file:
   master:/data1a/Subaru/SUPA/rerun/mineo-Abell1689/03430/W-S-I+/corr/wcs01098593.fits

then Wcs::_nWcsInfo becomes 2.

But WcsFormatter assumes that Wcs::_nWcsInfo is 1.

When the stacking program tries bcasting Wcs:
    - In serializing Wcs, the value _nWcsInfo = 2 is recorded and so read in
deserialization.
    - But in the deserialization, the formatter allocates only a single
element of _wcsInfo.

It causes inconsistency at the destructor, and SEGV arrises.
'''

The example file above has been copied and is used in the below test.
"""

import os
import os.path
import unittest
import pickle

import lsst.afw.image as afwImage
import lsst.utils.tests

testPath = os.path.abspath(os.path.dirname(__file__))

DATA = os.path.join(testPath, "data", "ticket2233.fits")


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
class WcsFormatterTest(unittest.TestCase):
    """Test the WCS formatter, by round-trip pickling."""

    def setUp(self):
        exposure = afwImage.ExposureF(DATA)
        self.wcs = exposure.getWcs()

    def tearDown(self):
        del self.wcs

    def testFormat(self):
        dumped = pickle.dumps(self.wcs)
        wcs = pickle.loads(dumped)
        self.assertEqual(wcs.getFitsMetadata().toString(), self.wcs.getFitsMetadata().toString())


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
