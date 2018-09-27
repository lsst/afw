#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2018 LSST Corporation.
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
"""Rewrite the headers of LSST reference catalogs to the correct AFW_TYPE.

We have many older reference catalogs saved as SourceCatalog, not
SimpleCatalog (identified via the AFW_TYPE header value in HDU1). This means
the FitsSchemaInputMapper cannot add the correct aliases for those files.
This script was written to correct the files in-place.
"""
from astropy.io import fits


def process_one(filename, update=False):
    with fits.open(filename) as hdulist:
        oldType = hdulist[1].header['AFW_TYPE']
        if oldType != "SOURCE":
            print("NOT updating (AFW_TYPE = {}): {}".format(oldType, filename))
            return

        print("Updating" if update else "Reading", filename, end=": ")
        print("Old AFW_TYPE =", oldType)
        if update:
            hdulist[1].header.set('AFW_TYPE', 'SIMPLE', 'Tells lsst::afw to load this as a Simple table.')
        hdulist.writeto(filename, overwrite=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+",
                        help="Files to rewrite the headers of.")
    parser.add_argument("--update", action="store_true",
                        help="Update the file headers in place (default just reads).")

    args = parser.parse_args()

    for filename in args.files:
        process_one(filename, update=args.update)


if __name__ == '__main__':
    main()
