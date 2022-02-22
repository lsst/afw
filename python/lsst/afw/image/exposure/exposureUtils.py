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

__all__ = ['bbox_to_convex_polygon', 'bbox_contains_sky_coords']

import numpy as np
import astropy.units as units
import lsst.geom


def bbox_to_convex_polygon(bbox, wcs, padding=10):
    """Convert a bounding box and wcs to a convex polygon on the sky, with paddding.

    The returned polygon has additional padding to ensure that the
    bounding box is entirely contained within it.  The default padding
    size was chosen to be sufficient for the most warped detectors at
    the edges of the HyperSuprimeCam focal plane.

    Parameters
    ----------
    bbox : `lsst.geom.Box2I`
        Bounding box to convert.
    wcs : `lsst.afw.image.SkyWcs`
        WCS associated with the bounding box.
    padding : `int`
       Pixel padding to ensure that bounding box is entirely contained
       within the resulting polygon.

    Returns
    -------
    convex_polygon : `lsst.sphgeom.ConvexPolygon`
       Will be None if wcs is not valid.
    """
    # Convert Box2I to Box2D, without modifying original.
    _bbox = lsst.geom.Box2D(bbox)
    _bbox.grow(padding)
    corners = [wcs.pixelToSky(corner).getVector()
               for corner in _bbox.getCorners()]
    return lsst.sphgeom.ConvexPolygon(corners)


def bbox_contains_sky_coords(bbox, wcs, ra, dec, padding=10):
    """Check if a set of sky positions are in the bounding box.

    This uses a two-step process: first check that the coordinates are
    inside a padded version of the bbox projected on the sky, and then
    project the remaining points onto the bbox, to avoid inverting
    the WCS outside of the valid region. The default padding
    size was chosen to be sufficient for the most warped detectors at
    the edges of the HyperSuprimeCam focal plane.

    Parameters
    ----------
    bbox : `lsst.geom.Box2I`
        Pixel bounding box to check sky positions in.
    wcs : `lsst.afw.image.SkyWcs`
        WCS associated with the bounding box.
    ra : `astropy.Quantity`, (N,)
        Array of Right Ascension, angular units.
    dec : `astropy.Quantity`, (N,)
        Array of Declination, angular units.
    padding : `int`
       Pixel padding to ensure that bounding box is entirely contained
       within the resulting polygon.

    Returns
    -------
    contained : `np.ndarray`, (N,)
       Boolean array indicating which points are contained in the
       bounding box.
    """
    poly = bbox_to_convex_polygon(bbox, wcs, padding=padding)

    _ra = np.atleast_1d(ra.to(units.radian).value).astype(np.float64)
    _dec = np.atleast_1d(dec.to(units.radian).value).astype(np.float64)

    radec_contained = poly.contains(_ra, _dec)

    x_in, y_in = wcs.skyToPixelArray(_ra, _dec, degrees=False)

    xy_contained = bbox.contains(x_in, y_in)

    return radec_contained & xy_contained
