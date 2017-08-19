#
# LSST Data Management System
# Copyright 2008-2013 LSST Corporation.
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

from __future__ import absolute_import, division, print_function
import unittest

from past.builtins import long
import numpy

import lsst.afw.image as afwImage
import lsst.utils.tests


class HeaderTestCase(lsst.utils.tests.TestCase):
    """Test that headers round-trip"""

    def testHeaders(self):
        header = {"STR": "String",
                  "INT": 12345,
                  "FLOAT": 678.9,
                  "NAN": numpy.nan,
                  "PLUSINF": numpy.inf,
                  "MINUSINF": -numpy.inf,
                  "LONG": long(987654321),
                  }

        exp = afwImage.ExposureI(0, 0)
        metadata = exp.getMetadata()
        for k, v in header.items():
            metadata.add(k, v)

        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            exp.writeFits(filename)
            exp = afwImage.ExposureI(filename)

        metadata = exp.getMetadata()
        for k, v in header.items():
            self.assertTrue(metadata.exists(k))
            if isinstance(v, float) and numpy.isnan(v):
                self.assertIsInstance(metadata.get(k), float)
                self.assertTrue(numpy.isnan(metadata.get(k)))
            else:
                self.assertEqual(metadata.get(k), v)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
