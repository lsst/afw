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

__all__ = ["normalize_lsst_images"]

import sys
import lsst.afw.image


def normalize_lsst_images(data, wcs=None):
    """Convert an `lsst.images` object to its afw equivalent for display.

    Parameters
    ----------
    data : `~typing.Any`
        Data to display.  If this is an `lsst.images.MaskedImage` (including
        subclasses such as `~lsst.images.VisitImage` and
        `~lsst.images.cells.CellCoadd`), `lsst.images.Image`, or
        `lsst.images.Mask`, it is converted to the corresponding
        `lsst.afw.image` type; anything else is returned unchanged.
    wcs : `lsst.afw.geom.SkyWcs` or `None`, optional
        WCS supplied by the caller.

    Returns
    -------
    data : `~typing.Any`
        The converted afw object, or the input unchanged.
    wcs : `lsst.afw.geom.SkyWcs` or `None`
        The WCS converted from ``data.sky_projection`` if present, else the
        caller-supplied ``wcs``.

    Raises
    ------
    RuntimeError
        Raised if ``wcs`` is not `None` and ``data`` has its own
        ``sky_projection``.

    Notes
    -----
    ``lsst.images`` is deliberately never imported here: it optionally
    depends on afw, so the dependency cannot go the other way.  If the
    module has never been imported, the caller cannot be holding one of
    its objects, so looking it up in `sys.modules` is sufficient.

    Only the image and mask planes are ever forwarded to display backends,
    so PSFs, variance, and other components are never serialized for
    display.  Mask schemas with more than 31 named planes cannot be
    represented as an `lsst.afw.image.Mask` and fail conversion.
    """
    images = sys.modules.get("lsst.images")
    if images is None or not isinstance(data, (images.MaskedImage, images.Image, images.Mask)):
        return data, wcs

    if data.sky_projection is not None:
        if wcs is not None:
            raise RuntimeError(
                "You may not specify a wcs with an lsst.images object that has its own sky_projection"
            )
        wcs = data.sky_projection.to_legacy()

    if isinstance(data, images.MaskedImage):
        data = lsst.afw.image.MaskedImage(
            data.image.to_legacy(),
            mask=data.mask.to_legacy(),
            variance=data.variance.to_legacy(),
            dtype=data.image.array.dtype,
        )
    else:
        # Image and Mask convert directly (Image as a zero-copy view).
        data = data.to_legacy()

    return data, wcs
