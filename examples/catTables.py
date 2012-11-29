#!/usr/bin/env python

import sys
import lsst.afw.table

def concatenate(inputFileNames):
    """Concatenate multiple catalogs (FITS tables from lsst.afw.table)"""
    inputCats = []
    size = 0
    table = None
    for inputFileName in inputFileNames:
        print "Reading %s" % inputFileName
        inputCat = lsst.afw.table.BaseCatalog.readFits(inputFileName)
        size += len(inputCat)
        if table is None:
            table = inputCat.table
        else:
            assert(table.schema == inputCat.table.schema)
        inputCats.append(inputCat)
    print "Allocating contiguous space for %d records" % size
    outputCat = lsst.afw.table.BaseCatalog(table.clone())
    outputCat.reserve(size)
    for inputFileName, inputCat in zip(inputFileNames, inputCats):
        print "Transferring records from %s" % inputFileName
        outputCat.extend(inputCat, deep=True)
    print "%d Records transferred" % len(outputCat)
    return outputCat

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "catTables.py: Concatenate multiple FITS tables (catalogs) from lsst.afw.table"
        print "Usage: catTables.py OUT IN1 IN2 [IN3...]"
        sys.exit(1)

    outName = sys.argv[1]
    catalogList = sys.argv[2:]
    out = concatenate(catalogList)
    out.writeFits(outName)
