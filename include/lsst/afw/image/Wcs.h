// -*- LSST-C++ -*-

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
 

#ifndef LSST_AFW_IMAGE_WCS_H
#define LSST_AFW_IMAGE_WCS_H


#include "Eigen/Core.h"
#include "lsst/base.h"
#include "lsst/daf/base.h"
#include "lsst/daf/data/LsstBase.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/deprecated.h"

struct wcsprm;                          // defined in wcs.h

namespace lsst {
    namespace daf {
        namespace base {
            class PropertySet;
        }
    }
namespace afw {
    namespace formatters {
        class WcsFormatter;
    }
namespace image {
    
/// 
/// @brief Implementation of the WCS standard for a any projection
/// 
/// Implements a single representation of the World Coordinate
/// System of a two dimensional image  The standard is defined in two papers
/// - Greisen & Calabretta, 2002 A&A 395, 1061
/// - Calabretta & Greisen, 2002, A&A 395, 1077
///  
/// In its simplest sense, Wcs is used to convert from position in the sky (in right ascension 
/// and declination) to pixel position on an image (and back again). It is, however, much more general 
/// than that and can understand a myriad of different coordinate systems.
/// 
/// A wcs can be constructed from a reference position (crval, crpix) and a translation matrix. Alternatively,
/// if you have the header from a fits file, you can create a Wcs object with the makeWcs() function. This
/// function determines whether your Wcs is one the subset of projection systems that is dealt with specially
/// by Lsst, and creates an object of the correct class. Otherwise, a pointer to a Wcs object is returned.
/// Most astronomical images use tangent plane projection, so makeWcs() returns a TanWcs object pointer
///
/// \code
/// import lsst.afw.image as afwImg
/// fitsHeader = afwImg.readMetadata(filename)
/// 
/// if 0:
///     #This doesn't work
///     wcs = afwImg.Wcs(fitsHeader)
///     
/// wcs = afwImg.makeWcs(fitsHeader)
/// 
/// pixelPosition = wcs.skyToPixel(ra, dec)
/// skyPosition = wcs.skyToPixel(xPosition, yPosition)
/// \endcode
/// 
/// 
/// o[
/// This class is implemented in by calls to the wcslib library
/// by Mark Calabretta http://www.atnf.csiro.au/people/mcalabre/WCS/
/// 
/// Note that we violate the Wcs standard in one minor way. The standard states that none
/// of the CRPIX or CRVAL keywords are required, for the header to be valid, and the appropriate values
/// should be set to 0.0 if the keywords are absent. This is a recipe for painful bugs in analysis, so
/// we violate the standard by insisting that the keywords CRPIX[1,2] and CRVAL[1,2] are present when
/// reading a header (keywords CRPIX1a etc are also accepted)

class Wcs : public lsst::daf::base::Persistable,
            public lsst::daf::data::LsstBase
{
public:
    typedef boost::shared_ptr<lsst::afw::image::Wcs> Ptr;
    typedef boost::shared_ptr<lsst::afw::image::Wcs const> ConstPtr;
    
    //Constructors
    Wcs();
    //Create a Wcs of the correct class using a fits header.
    friend Wcs::Ptr makeWcs(PTR(lsst::daf::base::PropertySet) fitsMetadata,
                            bool stripMetadata);

    Wcs(const lsst::afw::geom::PointD crval, const lsst::afw::geom::PointD crpix, const Eigen::Matrix2d &CD, 
        const std::string ctype1="RA---TAN", const std::string ctype2="DEC--TAN",
        double equinox=2000, std::string raDecSys="ICRS",
        const std::string cunits1="deg", const std::string cunits2="deg"
       );

    virtual ~Wcs();
    virtual Ptr clone(void) const;
    
    //Accessors
    lsst::afw::coord::Coord::Ptr getSkyOrigin() const;      //Return crval
    lsst::afw::geom::PointD getPixelOrigin() const;    //Return crpix
    Eigen::Matrix2d getCDMatrix() const;       //Return CD matrix
    
    virtual PTR(lsst::daf::base::PropertySet) getFitsMetadata() const;
    
    /// Return true iff Wcs is valid
    operator bool() const { return _nWcsInfo != 0; }
    
    bool isFlipped() const; //Does the Wcs follow the convention of North=Up, East=Left or not
    
    ///Sky area covered by a pixel at position \c pix00 in units of square degrees.
    double pixArea(lsst::afw::geom::PointD pix00) const;
    
    // Returns the pixel scale, in arcsec/pixel.
    double pixelScale() const;
    
    //Convert from raDec to pixel space. Formerly called raDecToXY() and
    //xyToRaDec(), but the name now reflects their increased generality. They may be
    //used, e.g. to convert xy to Galactic coordinates
    virtual lsst::afw::coord::Coord::Ptr pixelToSky(double pix1, double pix2) const;
    virtual lsst::afw::geom::PointD pixelToSky(double pix1, double pix2, bool) const;
    virtual lsst::afw::coord::Coord::Ptr pixelToSky(const lsst::afw::geom::PointD pixel) const;
    
    virtual lsst::afw::geom::PointD skyToPixel(double sky1, double sky2) const;
    virtual lsst::afw::geom::PointD skyToPixel(lsst::afw::coord::Coord::ConstPtr coord) const;
    lsst::afw::geom::PointD skyToIntermediateWorldCoord(lsst::afw::coord::Coord::ConstPtr coord) const;
    
    virtual bool hasDistortion() const {    return false;};
    
    lsst::afw::geom::LinearTransform getLinearTransform() const;
    
    lsst::afw::geom::AffineTransform linearizePixelToSky(
        lsst::afw::coord::Coord::ConstPtr const & coord,
        lsst::afw::coord::CoordUnit skyUnit = lsst::afw::coord::DEGREES
                                                        ) const;
    
    lsst::afw::geom::AffineTransform linearizePixelToSky(
        lsst::afw::geom::Point2D const & pix,
        lsst::afw::coord::CoordUnit skyUnit = lsst::afw::coord::DEGREES
                                                        ) const;

    lsst::afw::geom::AffineTransform linearizeSkyToPixel(
        lsst::afw::coord::Coord::ConstPtr const & coord,
            lsst::afw::coord::CoordUnit skyUnit = lsst::afw::coord::DEGREES
        ) const;
        
    lsst::afw::geom::AffineTransform linearizeSkyToPixel(
        lsst::afw::geom::Point2D const & pix,
        lsst::afw::coord::CoordUnit skyUnit = lsst::afw::coord::DEGREES
                                                        ) const;

    //Mutators
    void shiftReferencePixel(double dx, double dy); 

        
private:
    //Allow the formatter to access private goo
    LSST_PERSIST_FORMATTER(lsst::afw::formatters::WcsFormatter)
    
    void initWcsLib(const lsst::afw::geom::PointD crval, const lsst::afw::geom::PointD crpix,
                    const  Eigen::Matrix2d CD, 
                    const std::string ctype1, const std::string ctype2,
                    double equinox, std::string raDecSys,
                    const std::string cunits1, const std::string cunits2
                   );

    virtual void pixelToSkyImpl(double pixel1, double pixel2, double skyTmp[2]) const;

protected:

    //If you want to create a Wcs from a fits header, use makeWcs(). 
    //This is protected because the derived classes need to be able to see it.
    Wcs(PTR(lsst::daf::base::PropertySet) const fitsMetadata);
    
    Wcs(lsst::afw::image::Wcs const & rhs);
    Wcs& operator= (const Wcs &);        
    
    lsst::afw::coord::Coord::Ptr makeCorrectCoord(double sky0, double sky1) const;
    lsst::afw::geom::PointD convertCoordToSky(lsst::afw::coord::Coord::ConstPtr coord) const;
    
    virtual lsst::afw::geom::AffineTransform linearizePixelToSkyInternal(
                                                 lsst::afw::geom::Point2D const & pix,
                                                 lsst::afw::coord::Coord::ConstPtr const & coord,
                                                 lsst::afw::coord::CoordUnit skyUnit
                                                                        ) const;

    virtual lsst::afw::geom::AffineTransform linearizeSkyToPixelInternal(
                                                 lsst::afw::geom::Point2D const & pix,
                                                 lsst::afw::coord::Coord::ConstPtr const & coord,
                                                 lsst::afw::coord::CoordUnit skyUnit
                                                                        ) const;

    
    void initWcsLibFromFits(PTR(lsst::daf::base::PropertySet) const fitsMetadata);
    
    struct wcsprm* _wcsInfo;
    int _nWcsInfo;
    int _relax; ///< Degree of permissiveness for wcspih (0 for strict); see wcshdr.h for details.
    int _wcsfixCtrl; ///< Do potentially unsafe translations of non-standard unit strings? 0/1 = no/yes
    int _wcshdrCtrl; ///< Controls messages to stderr from wcshdr (0 for none); see wcshdr.h for details
    int _nReject;
};

namespace detail {
    PTR(lsst::daf::base::PropertySet)
    createTrivialWcsAsPropertySet(std::string const& wcsName, int const x0=0, int const y0=0);
    
    image::PointI getImageXY0FromMetadata(std::string const& wcsName, lsst::daf::base::PropertySet *metadata);
}

Wcs::Ptr makeWcs(PTR(lsst::daf::base::PropertySet) fitsMetadata, bool stripMetadata=false);
    
Wcs::Ptr makeWcs(lsst::afw::geom::PointD crval, lsst::afw::geom::PointD crpix,
                 double CD11, double CD12, double CD21, double CD22);
    
namespace detail {
    int stripWcsKeywords(PTR(lsst::daf::base::PropertySet) metadata, ///< Metadata to be stripped
                         CONST_PTR(Wcs) wcs                          ///< A Wcs with (implied) keywords
                        );
}
    
#if !defined(SWIG)
    extern Wcs NoWcs;
#endif
}}} // lsst::afw::image

#endif // LSST_AFW_IMAGE_WCS_H
