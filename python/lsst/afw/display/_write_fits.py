# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

__all__ = ["writeFitsImage"]

import io
import os
import subprocess

import lsst.afw.fits
import lsst.afw.geom
import lsst.afw.image
from lsst.daf.base import PropertyList, PropertySet
from lsst.geom import Extent2D


def _add_wcs(wcs_name: str, ps: PropertyList, x0: int = 0, y0: int = 0) -> None:
    ps.setInt(f"CRVAL1{wcs_name}", x0, "(output) Column pixel of Reference Pixel")
    ps.setInt(f"CRVAL2{wcs_name}", y0, "(output) Row pixel of Reference Pixel")
    ps.setDouble(f"CRPIX1{wcs_name}", 1.0, "Column Pixel Coordinate of Reference")
    ps.setDouble(f"CRPIX2{wcs_name}", 1.0, "Row Pixel Coordinate of Reference")
    ps.setString(f"CTYPE1{wcs_name}", "LINEAR", "Type of projection")
    ps.setString(f"CTYPE1{wcs_name}", "LINEAR", "Type of projection")
    ps.setString(f"CUNIT1{wcs_name}", "PIXEL", "Column unit")
    ps.setString(f"CUNIT2{wcs_name}", "PIXEL", "Row unit")


def writeFitsImage(
    file: str | int | io.BytesIO,
    data: lsst.afw.image.Image | lsst.afw.image.Mask,
    wcs: lsst.afw.geom.SkyWcs | None = None,
    title: str = "",
    metadata: PropertySet | None = None,
) -> None:
    """Write a simple FITS file with no extensions.

    Parameters
    ----------
    file : `str` or `int`
        Path to a file or a file descriptor.
    data : `lsst.afw.Image` or `lsst.afw.Mask`
        Data to be displayed.
    wcs : `lsst.afw.geom.SkyWcs` or `None`, optional
        WCS to be written to header to FITS file.
    title : `str`, optional
        If defined, the value to be stored in the ``OBJECT`` header.
        Overrides any value found in ``metadata``.
    metadata : `lsst.daf.base.PropertySet` or `None`, optional
        Additional information to be written to FITS header.
    """
    ps = PropertyList()

    # Seed with the external metadata, stripping wcs keywords.
    if metadata:
        lsst.afw.geom.stripWcsMetadata(metadata)
        ps.update(metadata)

    # Write WcsB, so that pixel (0,0) is correctly labelled (but ignoring XY0)
    _add_wcs("B", ps)

    if not wcs:
        _add_wcs("", ps)  # Works around a ds9 bug that WCSA/B is ignored if no WCS is present.
    else:
        shift = Extent2D(-data.getX0(), -data.getY0())
        if wcs.hasFitsApproximation():
            wcs = wcs.getFitsApproximation()
        new_wcs = wcs.copyAtShiftedPixelOrigin(shift)
        wcs_metadata = new_wcs.getFitsMetadata()
        ps.update(wcs_metadata)

    if title:
        ps.set("OBJECT", title, "Image being displayed")

    if isinstance(file, str):
        data.writeFits(file, metadata=ps)
    else:
        mem = lsst.afw.fits.MemFileManager()
        data.writeFits(manager=mem, metadata=ps)
        if isinstance(file, int):
            # Duplicate to prevent a double close, assuming the caller
            # will close the file descriptor they passed in.
            with os.fdopen(os.dup(file), "wb") as fh:
                fh.write(mem.getData())
        elif isinstance(file, subprocess.Popen):
            file.communicate(input=mem.getData())
        else:
            # Try the write() method directly.
            file.write(mem.getData())
