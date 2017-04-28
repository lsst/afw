#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
from past.builtins import basestring
import sys
import lsst.afw.table as afwTable
from functools import reduce


def concatenate(catalogList):
    """Concatenate multiple catalogs (FITS tables from lsst.afw.table)"""
    catalogList = [afwTable.BaseCatalog.readFits(c) if isinstance(
        c, basestring) else c for c in catalogList]

    schema = catalogList[0].schema
    for i, c in enumerate(catalogList[1:]):
        if c.schema != schema:
            raise RuntimeError("Schema for catalog %d not consistent" % (i+1))

    out = afwTable.BaseCatalog(schema)
    num = reduce(lambda n, c: n + len(c), catalogList, 0)
    out.preallocate(num)

    for catalog in catalogList:
        for record in catalog:
            out.append(out.table.copyRecord(record))

    return out


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "catTables.py: Concatenate multiple FITS tables (catalogs) from lsst.afw.table")
        print("Usage: catTables.py OUT IN1 IN2 [IN3...]")
        sys.exit(1)

    outName = sys.argv[1]
    catalogList = sys.argv[2:]
    out = concatenate(catalogList)
    out.writeFits(outName)
