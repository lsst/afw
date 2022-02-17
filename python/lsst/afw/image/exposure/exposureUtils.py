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
import lsst.geom


def bbox_to_convex_polygon(bbox, wcs, padding=10):
    """Convert a bounding box and wcs to a convex polygon, with paddding.

    The returned polygon has additional padding to ensure that the
    bounding box is entirely contained within it.

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
    if wcs is None:
        return None

    _bbox = lsst.geom.Box2D(bbox)
    _bbox.grow(padding)
    corners = [wcs.pixelToSky(corner).getVector()
               for corner in _bbox.getCorners()]
    return lsst.sphgeom.ConvexPolygon(corners)


def bbox_contains_sky_coords(bbox, wcs, ra, dec, degrees=False, padding=10):
    """Check if a set of sky positions are in the bounding box.

    This does a robust two-step procedure to avoid inverting the WCS outside
    of the valid region.

    Parameters
    ----------
    bbox : `lsst.geom.Box2I`
        Bounding box to convert.
    wcs : `lsst.afw.image.SkyWcs`
        WCS associated with the bounding box.
    ra : `np.ndarray`
        Array of Right Ascension.  Units are radians unless degrees=True.
    dec : `np.ndarray`
        Array of Declination.  Units are radians unless degrees=True.
    degrees : `bool`, optional
            Input ra, dec arrays are degrees if True.
    padding : `int`
       Pixel padding to ensure that bounding box is entirely contained
       within the resulting polygon.

    Returns
    -------
    contained : `np.ndarray`
       Boolean indicating which points are contained in the bounding box.

    Raises
    ------
    ValueError if exposure does not have a valid wcs.
    """
    poly = bbox_to_convex_polygon(bbox, wcs, padding=padding)

    if poly is None:
        raise ValueError("Exposure does not have a valid wcs.")

    if degrees:
        _ra = np.deg2rad(np.atleast_1d(ra)).astype(np.float64)
        _dec = np.deg2rad(np.atleast_1d(dec)).astype(np.float64)
    else:
        _ra = np.atleast_1d(ra).astype(np.float64)
        _dec = np.atleast_1d(dec).astype(np.float64)

    radec_contained = poly.contains(_ra, _dec)

    x_in, y_in = wcs.skyToPixelArray(_ra, _dec, degrees=False)

    xy_contained = ((x_in >= bbox.minX)
                    | (x_in <= bbox.maxX)
                    | (y_in >= bbox.minY)
                    | (y_in <= bbox.maxY))

    return radec_contained & xy_contained
