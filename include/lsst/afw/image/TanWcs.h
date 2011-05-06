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

#include "Eigen/Core.h"
#include "lsst/daf/base.h"
#include "lsst/daf/data/LsstBase.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/afw/image/Wcs.h" 
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"

struct wcsprm;                          // defined in wcs.h

namespace lsst {
namespace afw {
    namespace formatters {
        class TanWcsFormatter;
    }
namespace image {

/// 
/// @brief Implementation of the WCS standard for the special case of the Gnomonic 
/// (tangent plane) projection.
/// 
/// This class treats the special case of tangent plane projection. It extends the Wcs standard by 
/// optionally accounting for distortion in the image plane using the Simple Imaging Polynomial (SIP)
/// convention.
/// This convention is described in Shupe et al. (2005) (Astronomical Data Analysis Software and Systems
/// XIV, Asp Conf. Series Vol XXX, Ed: Shopbell et al.), and descibed in some more detail in
/// http://web.ipac.caltech.edu/staff/fmasci/home/wise/codeVdist.html
/// 
/// To convert from pixel coordintates to radec ("intermediate world coordinates"), first use the matrices
/// _sipA and _sipB to calculate undistorted coorinates (i.e where on the chip the image would lie if
/// the optics gave undistorted images), then pass these undistorted coorinates wcsp2s() to calculate radec.
/// 
/// For the reverse, radec -> pixels, convert the radec to undistorted coords, and then use the _sipAp and
/// _sipBp matrices to add in the distortion terms.
/// 
    class TanWcs : public lsst::afw::image::Wcs
{
    public:
        typedef boost::shared_ptr<lsst::afw::image::TanWcs> Ptr;    
        typedef boost::shared_ptr<lsst::afw::image::TanWcs const> ConstPtr;    

        //Constructors
        TanWcs();
        friend Wcs::Ptr makeWcs(lsst::daf::base::PropertySet::Ptr metadata, bool);
        TanWcs(const lsst::afw::geom::Point2D crval, const lsst::afw::geom::Point2D crpix, 
               const Eigen::Matrix2d &CD, 
               double equinox=2000, std::string raDecSys="FK5",
               const std::string cunits1="deg", const std::string cunits2="deg"
               );

        TanWcs(const lsst::afw::geom::Point2D crval, const lsst::afw::geom::Point2D crpix, 
               const Eigen::Matrix2d &CD, 
               Eigen::MatrixXd const & sipA, 
               Eigen::MatrixXd const & sipB, 
               Eigen::MatrixXd const & sipAp,
               Eigen::MatrixXd const & sipBp,  
               double equinox=2000, std::string raDecSys="FK5",
               const std::string cunits1="deg", const std::string cunits2="deg"
              );

        virtual ~TanWcs() {};
        
        virtual lsst::afw::image::Wcs::Ptr clone(void) const;

        bool operator==(const TanWcs &) const;

        // Returns the pixel scale, in arcsec/pixel.
        double pixelScale() const;
        
        // Applies the SIP AP and BP distortion (used in the skyToPixel direction)
        lsst::afw::geom::Point2D distortPixel(const lsst::afw::geom::Point2D pixel) const;
        // Applies the SIP A and B un-distortion (used in the pixelToSky direction)
        lsst::afw::geom::Point2D undistortPixel(const lsst::afw::geom::Point2D pixel) const;

        bool hasDistortion() const {    return _hasDistortion;};
        lsst::daf::base::PropertyList::Ptr getFitsMetadata() const;        
#if 0
        //Rely on base class implementation for now.
        lsst::afw::geom::AffineTransform linearizeAt(lsst::afw::geom::Point2D const & pix) const;
#endif        
        

        //Mutators
       //Because the base class provides the option of creating a Wcs without distortion coefficients
       //we supply a way of setting them here. This also help make code neater by breaking an
       //enormous constructor (above) into two small pieces 
       void setDistortionMatrices(Eigen::MatrixXd const & sipA, 
                                  Eigen::MatrixXd const & sipB,
                                  Eigen::MatrixXd const & sipAp,
                                  Eigen::MatrixXd const & sipBp
                                 );

    private:
        //If you want to create a TanWcs object from a fits header, use makeWcs()
        TanWcs(lsst::daf::base::PropertySet::Ptr const fitsMetadata);
        
        TanWcs(lsst::afw::image::TanWcs const & rhs);

        TanWcs & operator = (const TanWcs &);        

        virtual void pixelToSkyImpl(double pixel1, double pixel2, lsst::afw::geom::Angle skyTmp[2]) const;
        virtual lsst::afw::geom::Point2D skyToPixelImpl(lsst::afw::geom::Angle sky1, lsst::afw::geom::Angle sky2) const;

        //Allow the formatter to access private goo
        LSST_PERSIST_FORMATTER(lsst::afw::formatters::TanWcsFormatter)

        bool _hasDistortion;
        Eigen::MatrixXd _sipA, _sipB, _sipAp, _sipBp;
    
    };

}}}

#endif
