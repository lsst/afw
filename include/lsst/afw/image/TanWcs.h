// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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

#ifndef LSST_AFW_IMAGE_TANWCS_H
#define LSST_AFW_IMAGE_TANWCS_H

#include "Eigen/Core"
#include "lsst/daf/base/Citizen.h"
#include "lsst/daf/base/Persistable.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"

struct wcsprm;                          // defined in wcs.h

namespace lsst {
namespace daf {
    namespace base {
        class PropertySet;
    }
}
namespace afw {
    namespace formatters {
        class TanWcsFormatter;
    }
namespace image {

/**
 *  @brief Implementation of the WCS standard for the special case of the Gnomonic
 *  (tangent plane) projection.
 *
 *  This class treats the special case of tangent plane projection. It extends the Wcs standard by
 *  optionally accounting for distortion in the image plane using the Simple Imaging Polynomial (SIP)
 *  convention.
 *  This convention is described in Shupe et al. (2005) (Astronomical Data Analysis Software and Systems
 *  XIV, Asp Conf. Series Vol XXX, Ed: Shopbell et al.), and descibed in some more detail in
 *  http://web.ipac.caltech.edu/staff/fmasci/home/wise/codeVdist.html
 *
 *  To convert from pixel coordintates to radec ("intermediate world coordinates"), first use the matrices
 *  _sipA and _sipB to calculate undistorted coorinates (i.e where on the chip the image would lie if
 *  the optics gave undistorted images), then pass these undistorted coorinates wcsp2s() to calculate radec.
 *
 *  For the reverse, radec -> pixels, convert the radec to undistorted coords, and then use the _sipAp and
 *  _sipBp matrices to add in the distortion terms.
 */
class TanWcs : public afw::table::io::PersistableFacade<TanWcs>, public lsst::afw::image::Wcs {
public:
    /// Decode the SIP headers for a given matrix, if present.
    static void decodeSipHeader(
        daf::base::PropertySet const & fitsMetadata,
        std::string const & which,
        Eigen::MatrixXd & m
    );

    /**
     *  Construct a tangent plane wcs without distortion terms
     *
     *  @param crval    The sky position of the reference point
     *  @param crpix    The pixel position corresponding to crval in Lsst units
     *  @param cd       Matrix describing transformations from pixel to sky positions
     *  @param equinox  Equinox of coordinate system, eg 2000 (Julian) or 1950 (Besselian)
     *  @param raDecSys System used to describe right ascension or declination, e.g FK4, FK5 or ICRS
     *  @param cunits1  Units of sky position. One of deg, arcmin or arcsec
     *  @param cunits2  Units of sky position. One of deg, arcmin or arcsec
     */
    TanWcs(
        geom::Point2D const & crval, geom::Point2D const & crpix,
        Eigen::Matrix2d const & cd,
        double equinox=2000, std::string const & raDecSys="FK5",
        std::string const & cunits1="deg", std::string const & cunits2="deg"
    );

    /**
     *  Construct a tangent plane wcs with distortion terms
     *
     *  @param crval    The sky position of the reference point
     *  @param crpix    The pixel position corresponding to crval in Lsst units
     *  @param cd       Matrix describing transformations from pixel to sky positions
     *  @param sipA     Forward distortion matrix for axis 1
     *  @param sipB     Forward distortion matrix for axis 2
     *  @param sipAp    Reverse distortion matrix for axis 1
     *  @param sipBp    Reverse distortion matrix for axis 2
     *  @param equinox  Equinox of coordinate system, eg 2000 (Julian) or 1950 (Besselian)
     *  @param raDecSys System used to describe right ascension or declination, e.g FK4, FK5 or ICRS
     *  @param cunits1  Units of sky position. One of deg, arcmin or arcsec
     *  @param cunits2  Units of sky position. One of deg, arcmin or arcsec
     */
    TanWcs(
        geom::Point2D const & crval, geom::Point2D const & crpix,
        Eigen::Matrix2d const & cd,
        Eigen::MatrixXd const & sipA,
        Eigen::MatrixXd const & sipB,
        Eigen::MatrixXd const & sipAp,
        Eigen::MatrixXd const & sipBp,
        double equinox=2000, std::string const & raDecSys="FK5",
        std::string const & cunits1="deg", std::string const & cunits2="deg"
    );

    virtual ~TanWcs() {};

    /// Polymorphic deep-copy.
    std::shared_ptr<Wcs> clone() const override;

    /// Returns the pixel scale, in Angle/pixel.
    geom::Angle pixelScale() const;

    /// Applies the SIP AP and BP distortion (used in the skyToPixel direction)
    // NOTE that this accepts and returns FITS-style 1-indexed pixel coords, NOT LSST style 0-indexed
    geom::Point2D distortPixel(geom::Point2D const & pixel) const;

    /// Applies the SIP A and B un-distortion (used in the pixelToSky direction)
    // NOTE that this accepts and returns FITS-style 1-indexed pixel coords, NOT LSST style 0-indexed
    geom::Point2D undistortPixel(geom::Point2D const & pixel) const;

    bool hasDistortion() const override { return _hasDistortion;};

    void flipImage(int flipLR, int flipTB, lsst::afw::geom::Extent2I dimensions) const override;

    void rotateImageBy90(int nQuarter, lsst::afw::geom::Extent2I dimensions) const override;

    std::shared_ptr<daf::base::PropertyList> getFitsMetadata() const override;


    /**
     *  Set the distortion matrices
     *
     *  @param sipA  Forward distortion matrix for 1st axis
     *  @param sipB  Forward distortion matrix for 2nd axis
     *  @param sipAp Reverse distortion matrix for 1st axis
     *  @param sipBp Reverse distortion matrix for 2nd axis
     *
     *  Because the base class provides the option of creating a Wcs without distortion coefficients
     *  we supply a way of setting them here. This also help make code neater by breaking an
     *  enormous constructor (above) into two small pieces.
     */
    void setDistortionMatrices(
        Eigen::MatrixXd const & sipA,
        Eigen::MatrixXd const & sipB,
        Eigen::MatrixXd const & sipAp,
        Eigen::MatrixXd const & sipBp
    );

    /// Return the SIP forward distortion matrix for the 1st intermediate world coordinate system axis.
    Eigen::MatrixXd const & getSipA() const { return _sipA; }

    /// Return the SIP forward distortion matrix for the 2st intermediate world coordinate system axis.
    Eigen::MatrixXd const & getSipB() const { return _sipB; }

    /// Return the SIP reverse distortion matrix for x pixel coordinates.
    Eigen::MatrixXd const & getSipAp() const { return _sipAp; }

    /// Return the SIP reverse distortion matrix for y pixel coordinates.
    Eigen::MatrixXd const & getSipBp() const { return _sipBp; }

    /// Whether the object is persistable using afw::table::io archives.
    bool isPersistable() const override;

protected:

    TanWcs(TanWcs const & rhs);

    void pixelToSkyImpl(double pixel1, double pixel2, geom::Angle skyTmp[2]) const override;
    geom::Point2D skyToPixelImpl(geom::Angle sky1, geom::Angle sky2) const override;

private:

    friend std::shared_ptr<Wcs> makeWcs(std::shared_ptr<daf::base::PropertySet> const& metadata, bool);

    friend class TanWcsFactory;

    std::string getPersistenceName() const override;

    void write(OutputArchiveHandle & handle) const override;

    bool _isSubset(Wcs const &) const override;

    // Create an empty, invalid TanWcs.  Only used by TanWcsFormatter.
    TanWcs();

    /*
     *  Create a Wcs from a fits header.
     *
     *  Don't call this directly. Use makeWcs() instead, which will figure out which (if any)
     *  sub-class of Wcs is appropriate.
     */
    TanWcs(std::shared_ptr<daf::base::PropertySet const> const & fitsMetadata);

    TanWcs(afw::table::BaseRecord const & mainRecord, std::shared_ptr<afw::table::BaseRecord const> sipRecord);

    TanWcs & operator = (const TanWcs &);

    //Allow the formatter to access private goo
    LSST_PERSIST_FORMATTER(lsst::afw::formatters::TanWcsFormatter)

    bool _hasDistortion;
    Eigen::MatrixXd _sipA, _sipB, _sipAp, _sipBp;

};

}}} // namespace lsst::afw::image

#endif
