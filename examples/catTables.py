#!/usr/bin/env python

import sys
import lsst.afw.table as afwTable

def readCatalog(catalog):
    """Read a catalog into memory"""
    return afwTable.BaseCatalog.readFits(catalog) if isinstance(catalog, basestring) else catalog

def getCatalogRows(catalog, hdu=1):
    """Return number of rows in a catalog"""
    if not isinstance(catalog, basestring):
        return len(catalog)
    try:
        # I believe this is significantly faster than reading the whole thing
        import pyfits
        fits = pyfits.open(catalog)
        try:
            rows = len(fits[hdu].data)
            return rows
        finally:
            fits.close()
    except ImportError:
        # We have to read the whole thing just to get the length
        return len(afwTable.BaseCatalog.readFits(catalog))

def concatenate(catalogList, doPreallocate=True):
    """Concatenate multiple catalogs (FITS tables from lsst.afw.table)"""

    if doPreallocate:
        num = sum(getCatalogRows(cat) for cat in catalogList)
        print "%d rows from %d catalogs" % (num, len(catalogList))

    cat = readCatalog(catalogList[0])
    schema = cat.schema

    out = afwTable.BaseCatalog(schema)
    table = out.table

    if doPreallocate:
        table.preallocate(num)
        out.reserve(num)

    for i, cat in enumerate(catalogList):
        print cat
        cat = readCatalog(cat)
        if cat.schema != schema:
            raise RuntimeError("Schema for catalog %d not consistent" % (i+1))
        for record in cat:
            out.append(table.copyRecord(record))
        del cat

    return out


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "catTables.py: Concatenate multiple FITS tables (catalogs) from lsst.afw.table"
        print "Usage: catTables.py OUT IN1 IN2 [IN3...]"
        sys.exit(1)

    outName = sys.argv[1]
    catalogList = sys.argv[2:]
    out = concatenate(catalogList)
    out.writeFits(outName)
