from __future__ import annotations

__all__ = ["writeFitsImage"]

import os
import subprocess

from lsst.daf.base import PropertyList, PropertySet
from lsst.afw.geom import stripWcsMetadata
from lsst.geom import Extent2D
import lsst.afw.fits


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
    file: str | int, data, wcs=None, title: str = "", metadata: PropertySet | None = None
) -> None:
    """Write a simple FITS file with no extensions.

    Parameters
    ----------
    file : `str` or `int`
        Path to a file or a file descriptor.
    data : `lsst.afw.Image` or `lsst.afw.Mask`
        Data to be displayed.
    wcs : `lsst.afw.geomSkyWcs` or `None`
        WCS to be written to header to FITS file.
    title : `str`
        If defined, the value to be stored in the ``OBJECT`` header.
        Overrides ny value found in ``metadata``.
    metadata : `lsst.daf.base.PropertySet` or `None`
        Additional information to be written to FITS header.
    """
    ps = PropertyList()

    # Seed with the external metadata, stripping wcs keywords.
    if metadata:
        print("STRIP")
        stripWcsMetadata(metadata)
        print("METADATA COPY")
        ps.update(metadata)

    # Generate WcsA, pixel coordinates, allowing for X0 and Y0.
    # Automatically written by writeFits below
    # _add_wcs("A", ps, data.getX0(), data.getY0())

    # Now WcsB, so that pixel (0,0) is correctly labelled (but ignoring XY0)
    # Ignored by writeFits below
    _add_wcs("B", ps)

    if not wcs:
        _add_wcs("", ps)  # Works around a ds9 bug that WCSA/B is ignored if no WCS is present.
    else:
        shift = Extent2D(-data.getX0(), -data.getY0())
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
            with os.fdopen(file, "wb") as fh:
                fh.write(mem.getData())
        elif isinstance(file, subprocess.Popen):
            file.communicate(input=mem.getData())
        else:
            # Try the write() method directly.
            file.write(mem.getData())
