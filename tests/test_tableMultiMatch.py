# LSST Data Management System
# Copyright 2016 LSST Corporation.
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
# The classes in this test are a little non-standard to reduce code
# duplication and support automated unittest discovery.
# A base class includes all the code that implements the testing and
# itself inherits from unittest.TestCase. unittest automated discovery
# will scan all classes that inherit from unittest.TestCase and invoke
# any test methods found. To prevent this base class from being executed
# the test methods are placed in a different class that does not inherit
# from unittest.TestCase. The actual test classes then inherit from
# both the testing class and the implementation class allowing test
# discovery to only run tests found in the subclasses.
"""Tests for lsst.afw.table.multiMatch."""

import os
import re
import unittest

import numpy as np

import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom
import lsst.pex.exceptions as pexExcept
import lsst.utils
import lsst.utils.tests


try:
    afwdataDir = lsst.utils.getPackageDir("afwdata")
except pexExcept.NotFoundError:
    afwdataDir = None


class TestGroupView(lsst.utils.tests.TestCase):
    """Test case for lsst.afw.table.multiMatch.GroupView."""

    def setUp(self):
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.schema.addField("flux_flux", type=np.float64)
        self.schema.addField("flux_fluxSigma", type=np.float64)
        self.schema.addField("flux_flag", type="Flag")
        self.table = afwTable.SourceTable.make(self.schema)
        self.table.definePsfFlux("flux")

        band = 2  # SDSS r

        # Read SDSS catalogue
        with open(os.path.join(afwdataDir, "CFHT", "D2", "sdss.dat"), "r") as ifd:
            sdss = afwTable.SourceCatalog(self.table)

            PRIMARY = 1  # values of mode

            for line in ifd.readlines():
                if re.search(r"^\s*#", line):
                    continue

                fields = line.split()
                objId = int(fields[0])
                fields[1]
                mode = int(fields[2])
                ra, dec = [float(f) for f in fields[3:5]]
                psfMags = [float(f) for f in fields[5:]]

                if mode == PRIMARY:
                    s = sdss.addNew()

                s.setId(objId)
                s.setRa(ra * afwGeom.degrees)
                s.setDec(dec * afwGeom.degrees)
                s.set(self.table.getPsfFluxKey(), psfMags[band])

        # Read catalalogue built from the template image
        # Read SDSS catalogue
        with open(os.path.join(afwdataDir, "CFHT", "D2", "template.dat"), "r") as ifd:
            template = afwTable.SourceCatalog(self.table)

            for line in ifd.readlines():
                if re.search(r"^\s*#", line):
                    continue

                fields = line.split()
                id_, flags = [int(f) for f in fields[0:2]]
                ra, dec = [float(f) for f in fields[2:4]]
                flux = [float(f) for f in fields[4:]]

                if flags & 0x1:             # EDGE
                    continue

                s = template.addNew()
                s.setId(id_)
                s.set(afwTable.SourceTable.getCoordKey().getRa(),
                      ra * afwGeom.degrees)
                s.set(afwTable.SourceTable.getCoordKey().getDec(),
                      dec * afwGeom.degrees)
                s.set(self.table.getPsfFluxKey(), flux[0])

        m = afwTable.MultiMatch(self.schema,
                                dict(visit=np.int64),
                                RecordClass=afwTable.SimpleRecord)
        m.add(sdss, {'visit': 1})
        m.add(template, {'visit': 2})

        self.matchedCatalog = m.finish()

    def tearDown(self):
        del self.table
        del self.schema
        del self.matchedCatalog

    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
    def testGroupViewBuild(self):
        """Simple test of building a GroupView from a MultiMatch. See DM-8557.

        Table creation is copied from testSourceMatch.py's
        SourceMatchTestCase.testPhotometricCalib().
        """
        allMatches = afwTable.GroupView.build(self.matchedCatalog)
        self.assertTrue(len(allMatches) > 0)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
