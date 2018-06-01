#
# LSST Data Management System
# Copyright 2017 LSST Corporation.
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

'''
Test for the custom __dir__ added to tables/baseContinued.py
'''
import unittest
import lsst.utils.tests

import lsst.geom
import lsst.afw.table


class DirTestCase(lsst.utils.tests.TestCase):
    def testDir(self):
        '''Ensure the custom __dir__ returns all the expected values'''
        # Create a source catalog with a minimal schema
        schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        catalog = lsst.afw.table.SourceCatalog(schema)
        record = catalog.addNew()
        record['coord_dec'] = lsst.geom.degrees*(-5.0)
        record['coord_ra'] = lsst.geom.degrees*(22)
        record['id'] = 8
        record['parent'] = 3
        # Compare catalog attributes with those from various catalog subclasses
        attrNames = dir(catalog)
        desiredNames = set(['_columns', '__module__', 'getX', 'getY',
                            'asAstropy', 'hasPsfFluxSlot'])
        self.assertTrue(desiredNames.issubset(attrNames))


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == '__main__':
    lsst.utils.tests.init()
    unittest.main()
