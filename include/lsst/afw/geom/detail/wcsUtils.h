/*
 * LSST Data Management System
 * Copyright 2017 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#ifndef LSST_AFW_GEOM_DETAILS_WCSUTILS_H
#define LSST_AFW_GEOM_DETAILS_WCSUTILS_H

#include <memory>
#include <string>

#include "Eigen/Core"

#include "lsst/daf/base.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/SkyWcs.h"

namespace lsst {
namespace afw {
namespace geom {
namespace detail {

/**
 * @internal Define a trivial named WCS that maps the lower left corner pixel of an image to its parent
 * coordinate.
 *
 * Sets the following keywords:
 * - CRPIX[12]<wcsName> = 1
 * - CRVAL[12]<wcsName> = xy0
 * - CTYPE[12]<wcsName> = "LINEAR"
 * - CUNIT[12]<wcsName> = "PIXEL"
 *
 * @param wcsName WCS name, e.g. "A"
 * @param xy0  Parent coordinate of lower left corner pixel of image
 */
std::shared_ptr<daf::base::PropertyList> createTrivialWcsAsPropertySet(std::string const& wcsName,
                                                                       geom::Point2I const& xy0);

/**
 * @internal Delete metadata for a named WCS
 *
 * Delete keywords created by createTrivialWcsAsPropertySet plus CD keywords and WCSAXES.
 * Missing entries are silently ignored.
 *
 * @param[in,out] metadata The metadata
 * @param[in] wcsName The WCS to search, e.g. "A"
 */
void deleteBasicWcsMetadata(daf::base::PropertySet& metadata, std::string const& wcsName);

/**
 * Read a CD matrix from FITS WCS metadata
 *
 * The elements of the returned matrix are in degrees
 *
 * @throws pex::exceptions::InvalidParameterError if no CD matrix coefficients found
 * (missing coefficients are set to 0, as usual, but they cannot all be missing).
 */
Eigen::Matrix2d getCdMatrixFromMetadata(daf::base::PropertySet& metadata);

/**
 * @internal Return XY0 as specified by a trivial named WCS, and delete the WCS keywords, if present.
 *
 * If CRPIX[12]<wcsName> are both present and both equal to 1,
 * and CRVAL[12]<wcsName> are both present, then set XY0 = CRVAL12<wcsName>
 * and delete all entries for this WCS. Otherwise silently return (0, 0) and delete nothing.
 *
 * Note: if any other keywords that should be present for the named WCS are missing,
 * they are silently ignored.
 *
 * @param[in,out] metadata the metadata, maybe containing the WCS
 * @param[in] wcsName the WCS to search, e.g. "A"
 * @param[in] strip  If true, strip the WCS keywords using deleteBasicWcsMetadata, if present.
 */
geom::Point2I getImageXY0FromMetadata(daf::base::PropertySet& metadata, std::string const& wcsName,
                                      bool strip = false);

/**
 * @internal Extract a SIP matrix from FITS TAN-SIP WCS metadata
 *
 * @param[in] metadata  FITS metadata
 * @param[in] name  Name of TAN-SIP matrix; one of A, B, Ap, Bp
 * @returns the specified TAN-SIP matrix
 */
Eigen::MatrixXd getSipMatrixFromMetadata(daf::base::PropertySet const& metadata, std::string const& name);

/**
 * @internal Return True if the metadata includes data for the specified FITS TAN-SIP WCS matrix
 *
 * @param[in] metadata  FITS metadata
 * @param[in] name  Name of TAN-SIP matrix; one of A, B, Ap, Bp
 *
 * Checks for name_ORDER, e.g. AP_ORDER
 */
bool hasSipMatrix(daf::base::PropertySet const& metadata, std::string const& name);

/**
 * @internal Encode a SIP matrix as FITS TAN-SIP WCS metadata
 *
 * @param[in] m  SIP matrix; must be square and at least 1 x 1
 * @param[in] name  Name of SIP matrix; one of A, B, Ap, Bp
 * @returns Metadata for SIP matrix:
 *    name_ORDER = order
 *    name_i_j = m(i, j) for each i, j
 */
std::shared_ptr<daf::base::PropertyList> makeSipMatrixMetadata(Eigen::MatrixXd const& matrix,
                                                               std::string const& name);

/**
 * Make metadata for a TAN-SIP WCS without inverse matrices
 *
 * @param[in] crpix    Center of projection on the CCD using the LSST convention:
 *                      0, 0 is the lower left pixel of the parent image
 * @param[in] crval    Center of projection on the sky
 * @param[in] cdMatrix CD matrix, where element (i-1, j-1) corresponds to FITS keyword CDi_j
 *                      and i, j have range [1, 2]. May be computed by calling makeCdMatrix.
 * @param[in] sipA     Forward distortion matrix for axis 1
 * @param[in] sipB     Forward distortion matrix for axis 2
 */
std::shared_ptr<daf::base::PropertyList> makeTanSipMetadata(Point2D const& crpix,
                                                            coord::IcrsCoord const& crval,
                                                            Eigen::Matrix2d const& cdMatrix,
                                                            Eigen::MatrixXd const& sipA,
                                                            Eigen::MatrixXd const& sipB);

/**
 * Make metadata for a TAN-SIP WCS
 *
 * @param[in] crpix    Center of projection on the CCD using the LSST convention:
 *                      0, 0 is the lower left pixel of the parent image
 * @param[in] crval    Center of projection on the sky
 * @param[in] cdMatrix CD matrix, where element (i-1, j-1) corresponds to FITS keyword CDi_j
 *                      and i, j have range [1, 2]. May be computed by calling makeCdMatrix.
 * @param[in] sipA     Forward distortion matrix for axis 1
 * @param[in] sipB     Forward distortion matrix for axis 2
 * @param[in] sipAp    Reverse distortion matrix for axis 1
 * @param[in] sipBp    Reverse distortion matrix for axis 2
 */
std::shared_ptr<daf::base::PropertyList> makeTanSipMetadata(
        Point2D const& crpix, coord::IcrsCoord const& crval, Eigen::Matrix2d const& cdMatrix,
        Eigen::MatrixXd const& sipA, Eigen::MatrixXd const& sipB, Eigen::MatrixXd const& sipAp,
        Eigen::MatrixXd const& sipBp);

}  // namespace detail
}  // namespace geom
}  // namespace afw
}  // namespace lsst

#endif  // LSST_AFW_GEOM_DETAILS_WCSUTILS_H
