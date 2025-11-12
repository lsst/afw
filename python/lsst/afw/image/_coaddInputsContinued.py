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

from lsst.utils import continueClass

from ._imageLib import CoaddInputs

import numpy as np

__all__ = []  # import this module only for its side effects


@continueClass
class CoaddInputs:  # noqa: F811

    def subset_containing_ccds(self, point, wcs, includeValidPolygon=False):
        """Return a view (shallow copy) of ExposureCatalog containing only the
        subset of detectors that contain the given point.

        Parameters
        ----------
        point : `~lsst.geom.Point2D`
            Point in the coadd coordinate system.
        wcs : `lsst.geom.SkyWcs`
            WCS for the coadd coordinate system. This is ignored if the
            CoaddInputs are made by stitching cell_coadds.
        includeValidPolygon : `bool`, optional
            If True, check that the point is within the validPolygon of those records which have one.

        Returns
        -------
        subset : `~lsst.afw.table.ExposureCatalog`
            ExposureCatalog containing only the relevant detector records.
        """

        ccds = self.ccds
        # If the records have a WCS attached, we interpret that to mean that
        # they come from a genuine afw exposure. If not, we interpret that to
        # mean they come from cell_coadds. For the latter, the validPolygons
        # are already in coadd coordinates and WCS lookup is not needed.
        if len(ccds) == 0 or ccds[0].wcs is not None:
            return ccds.subsetContaining(point, wcs, includeValidPolygon)
        else:
            cuts = np.array([record.validPolygon.contains(point) for record in ccds])
            return ccds[cuts]

    def subset_containing_visits(self, point, wcs, includeValidPolygon=False):
        """Return a view (shallow copy) of ExposureCatalog containing only the
        subset of visits that contain the given point.

        Parameters
        ----------
        point : `~lsst.geom.Point2D`
            Point in the coadd coordinate system.
        wcs : `lsst.geom.SkyWcs`
            WCS for the coadd coordinate system. This is ignored if the
            CoaddInputs are made by stitching cell_coadds.
        includeValidPolygon : `bool`, optional
            If True, check that the point is within the validPolygon of those records which have one.

        Returns
        -------
        subset : `~lsst.afw.table.ExposureCatalog`
            ExposureCatalog containing only the relevant visit records.
        """

        visits = self.visits
        if len(visits) == 0 or visits[0].wcs is not None:
            return visits.subsetContaining(point, wcs, includeValidPolygon)
        else:
            ccd_cuts = np.array([record.validPolygon.contains(point) for record in self.ccds])
            visit_cuts = np.isin(visits["visit"], self.ccds["visit"][ccd_cuts])
            return visits[visit_cuts]
