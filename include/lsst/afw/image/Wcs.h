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

#include <limits>
#include "Eigen/Core"
#include "lsst/base.h"
#include "lsst/daf/base/Persistable.h"
#include "lsst/daf/base/Citizen.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/XYTransform.h"
#include "lsst/afw/table/io/Persistable.h"

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
    namespace table {
        class BaseRecord;
    }
namespace image {

///
/// Implementation of the WCS standard for a any projection
///
/// Implements a single representation of the World Coordinate
/// System of a two dimensional image  The standard is defined in two papers
/// - Greisen & Calabretta, 2002 A&A 395, 1061
/// - Calabretta & Greisen, 2002, A&A 395, 1077
///
/// In its simplest sense, Wcs is used to convert from position in the sky (in
/// right ascension and declination) to pixel position on an image (and back
/// again). It is, however, much more general than that and can understand a
/// myriad of different coordinate systems.
///
/// A wcs can be constructed from a reference position (crval, crpix) and a
/// translation matrix. Alternatively, if you have the header from a FITS file,
/// you can create a Wcs object with the makeWcs() function. This function
/// determines whether your Wcs is one the subset of projection systems that is
/// dealt with specially by LSST, and creates an object of the correct
/// class. Otherwise, a pointer to a Wcs object is returned.  Most astronomical
/// images use tangent plane projection, so makeWcs() returns a TanWcs object
/// pointer
///
///     import lsst.afw.image as afwImg
///     fitsHeader = afwImg.readMetadata(filename)
///
///     if 0:
///         #This doesn't work
///         wcs = afwImg.Wcs(fitsHeader)
///
///     wcs = afwImg.makeWcs(fitsHeader)
///
///     pixelPosition = wcs.skyToPixel(ra, dec)
///     skyPosition = wcs.skyToPixel(xPosition, yPosition)
///
///
/// o[
/// This class is implemented in by calls to the wcslib library
/// by Mark Calabretta http://www.atnf.csiro.au/people/mcalabre/WCS/
///
/// Note that we violate the WCS standard in one minor way. The standard states
/// that none of the CRPIX or CRVAL keywords are required, for the header to be
/// valid, and the appropriate values should be set to 0.0 if the keywords are
/// absent. This is a recipe for painful bugs in analysis, so we violate the
/// standard by insisting that the keywords CRPIX[1,2] and CRVAL[1,2] are
/// present when reading a header (keywords CRPIX1a etc are also accepted)

class Wcs : public lsst::daf::base::Persistable,
            public lsst::daf::base::Citizen,
            public afw::table::io::PersistableFacade<Wcs>,
            public afw::table::io::Persistable
{
public:
    typedef std::shared_ptr<Wcs> Ptr;
    typedef std::shared_ptr<Wcs const> ConstPtr;

    /**
     *  Create a Wcs of the correct class using a FITS header.
     *
     *  Set stripMetadata=true to remove processed keywords from the PropertySet.
     */
    friend PTR(Wcs) makeWcs(PTR(lsst::daf::base::PropertySet) const& fitsMetadata,
                            bool stripMetadata);

    /** Create a Wcs object with some known information.
     *
     * @param crval The sky position of the reference point
     * @param crpix The pixel position corresponding to crval in LSST units
     * @param CD    Matrix describing transformations from pixel to sky positions
     * @param ctype1 Projection system used (see description of Wcs)
     * @param ctype2 Projection system used (see description of Wcs)
     * @param equinox Equinox of coordinate system, eg 2000 (Julian) or 1950 (Besselian)
     * @param raDecSys System used to describe right ascension or declination, e.g FK4, FK5 or ICRS
     * @param cunits1 Units of sky position. One of deg, arcmin or arcsec
     * @param cunits2 Units of sky position. One of deg, arcmin or arcsec
     *
     *@note LSST units are zero indexed while FITs units are 1 indexed. So a value of crpix stored in a fits
     * header of 127,127 corresponds to a pixel position in LSST units of 128, 128
     */
    Wcs(lsst::afw::geom::Point2D const & crval, lsst::afw::geom::Point2D const & crpix,
        Eigen::Matrix2d const & CD,
        std::string const & ctype1="RA---TAN", std::string const & ctype2="DEC--TAN",
        double equinox=2000, std::string const & raDecSys="ICRS",
        std::string const & cunits1="deg", std::string const & cunits2="deg"
       );

    virtual ~Wcs();
    virtual Ptr clone(void) const;

    bool operator==(Wcs const & other) const;
    bool operator!=(Wcs const & other) const { return !(*this == other); }

    /// Returns CRVAL. This need not be the centre of the image.
    PTR(lsst::afw::coord::Coord) getSkyOrigin() const;

    /// Returns CRPIX (corrected to LSST convention).
    lsst::afw::geom::Point2D getPixelOrigin() const;

    /// Returns the CD matrix.
    Eigen::Matrix2d getCDMatrix() const;

    /// Flip CD matrix around the y-axis
    virtual void flipImage(int flipLR, int flipTB, lsst::afw::geom::Extent2I dimensions) const;

    /// Rotate image by nQuarter times 90 degrees.
    virtual void rotateImageBy90(int nQuarter, lsst::afw::geom::Extent2I dimensions) const;

    /// Return a PropertyList containing FITS header keywords that can be used to save the Wcs.x
    virtual PTR(lsst::daf::base::PropertyList) getFitsMetadata() const;

    /**
     *  Does the Wcs follow the convention of North=Up, East=Left?
     *
     *  The conventional sense for a WCS image is to have North up and East to the left, or at least to be
     *  able to rotate the image to that orientation. It is possible to create a "flipped" WCS, where East
     *  points right when the image is rotated such that North is up. Flipping a WCS is akin to producing a
     *  mirror image. This function tests whether the image is flipped or not.
     */
    bool isFlipped() const;

    /** Sky area covered by a pixel at position `pix00` in units of square degrees.
     *
     * @param pix00 The pixel point where the area is desired
     */
    double pixArea(lsst::afw::geom::Point2D pix00) const;

    /// Returns the pixel scale [Angle/pixel]
    geom::Angle pixelScale() const;

    /**
     *  Convert from pixel position to sky coordinates (e.g. RA/dec)
     *
     *  Convert a pixel position (e.g. x,y) to a celestial coordinate (e.g. RA/dec). The output coordinate
     *  system depends on the values of CTYPE used to construct the object. For RA/dec, the CTYPES should
     *  be RA---TAN and DEC--TAN.
     */
    PTR(coord::Coord) pixelToSky(double pix1, double pix2) const;

    /**
     *  Convert from pixel position to sky coordinates (e.g. RA/dec)
     *
     *  Convert a pixel position (e.g. x,y) to a celestial coordinate (e.g. RA/dec). The output coordinate
     *  system depends on the values of CTYPE used to construct the object. For RA/dec, the CTYPES should
     *  be RA---TAN and DEC--TAN.
     */
    PTR(coord::Coord) pixelToSky(lsst::afw::geom::Point2D const & pixel) const;

    /**
     *  Convert from pixel position to sky coordinates (e.g. RA/dec)
     *
     *  @note This routine is designed for the knowledgeable user in need of performance;
     *  it's safer to call the version that returns a PTR(Coord).
     */
    void pixelToSky(
        double pixel1, double pixel2, geom::Angle& sky1, geom::Angle& sky2
    ) const;

    /**
     *  Convert from sky coordinates (e.g. RA/dec) to pixel positions
     *
     *  Convert a sky position (e.g. RA/dec) to a pixel position. The exact meaning of sky1, sky2
     *  and the return value depend on the properties of the wcs (i.e. the values of CTYPE1 and
     *  CTYPE2), but the inputs are usually RA/dec. The outputs are x and y pixel position.
     *
     *  ASSUMES the angles are in the appropriate coordinate system for this Wcs.
     */
    geom::Point2D skyToPixel(geom::Angle sky1, geom::Angle sky2) const;

    /// Convert from sky coordinates (e.g. RA/dec) to pixel positions.
    geom::Point2D skyToPixel(coord::Coord const & coord) const;

    /**
     *  Convert from sky coordinates (e.g. RA/dec) to intermediate world coordinates
     *
     *  Intermediate world coordinates are in DEGREES.
     */
    geom::Point2D skyToIntermediateWorldCoord(coord::Coord const & coord) const;

    virtual bool hasDistortion() const {    return false;};

    afw::coord::CoordSystem getCoordSystem() const { return _coordSystem; };

    double getEquinox() const;

    /**
     * Return true if a WCS has the same coordinate system and equinox as this one
     *
     * There are two special cases:
     * - Equinox is ignored if the coordinate system is ICRS
     * - FK5 J2000 is considered the same as ICRS
     */
    bool isSameSkySystem(Wcs const &wcs) const;

    /**
     * Return the linear part of the Wcs, the CD matrix in FITS-speak, as an AffineTransform.
     */
    geom::LinearTransform getLinearTransform() const;

    /**
     * Return the local linear approximation to Wcs::pixelToSky at a point given in sky coordinates.
     *
     * The local linear approximation is defined such the following is true (ignoring floating-point errors):
     *
     *     wcs.linearizePixelToSky(sky, skyUnit)(wcs.skyToPixel(sky)) == sky.getPosition(skyUnit);
     *
     * (recall that AffineTransform::operator() is matrix multiplication with the augmented point (x,y,1)).
     *
     * This is currently implemented as a numerical derivative, but we should specialise the Wcs class
     * (or rather its implementation) to handle "simple" cases such as TAN-SIP analytically
     *
     * @param[in] coord   Position in sky coordinates where transform is desired.
     * @param[in] skyUnit Units to use for sky coordinates; units of matrix elements will be skyUnits/pixel.
     */
    geom::AffineTransform linearizePixelToSky(
        coord::Coord const & coord,
        geom::AngleUnit skyUnit = geom::degrees
    ) const;

    /**
     * Return the local linear approximation to Wcs::pixelToSky at a point given in pixel coordinates.
     *
     * The local linear approximation is defined such the following is true (ignoring floating-point errors):
     *
     *     wcs.linearizePixelToSky(pix, skyUnit)(pix) == wcs.pixelToSky(pix).getPosition(skyUnit)
     *
     * (recall that AffineTransform::operator() is matrix multiplication with the augmented point (x,y,1)).
     *
     * This is currently implemented as a numerical derivative, but we should specialise the Wcs class
     * (or rather its implementation) to handle "simple" cases such as TAN-SIP analytically
     *
     * @param[in] pix     Position in pixel coordinates where transform is desired.
     * @param[in] skyUnit Units to use for sky coordinates; units of matrix elements will be skyUnits/pixel.
     */
    geom::AffineTransform linearizePixelToSky(
        geom::Point2D const & pix,
        geom::AngleUnit skyUnit = geom::degrees
    ) const;

    /**
     * Return the local linear approximation to Wcs::skyToPixel at a point given in sky coordinates.
     *
     * The local linear approximation is defined such the following is true (ignoring floating-point errors):
     *
     *     wcs.linearizeSkyToPixel(sky, skyUnit)(sky.getPosition(skyUnit)) == wcs.skyToPixel(sky)
     *
     * (recall that AffineTransform::operator() is matrix multiplication with the augmented point (x,y,1)).
     *
     * This is currently implemented as a numerical derivative, but we should specialise the Wcs class
     * (or rather its implementation) to handle "simple" cases such as TAN-SIP analytically
     *
     * @param[in] coord   Position in sky coordinates where transform is desired.
     * @param[in] skyUnit Units to use for sky coordinates; units of matrix elements will be pixels/skyUnit.
     */
    geom::AffineTransform linearizeSkyToPixel(
        coord::Coord const & coord,
        geom::AngleUnit skyUnit = geom::degrees
    ) const;

    /**
     * Return the local linear approximation to Wcs::skyToPixel at a point given in pixel coordinates.
     *
     * The local linear approximation is defined such the following is true (ignoring floating-point errors):
     *
     *     wcs.linearizeSkyToPixel(pix, skyUnit)(wcs.pixelToSky(pix).getPosition(skyUnit)) == pix
     *
     * (recall that AffineTransform::operator() is matrix multiplication with the augmented point (x,y,1)).
     *
     * This is currently implemented as a numerical derivative, but we should specialise the Wcs class
     * (or rather its implementation) to handle "simple" cases such as TAN-SIP analytically
     *
     * @param[in] pix     Position in pixel coordinates where transform is desired.
     * @param[in] skyUnit Units to use for sky coordinates; units of matrix elements will be pixels/skyUnit.
     */
    geom::AffineTransform linearizeSkyToPixel(
        geom::Point2D const & pix,
        geom::AngleUnit skyUnit = geom::degrees
    ) const;

    // Mutators; the first one is virtual, even though it will never be overridden,
    // to make sure subclasses use the correct version of both

    /**
     *  Move the pixel reference position by (dx, dy)
     *
     *  Used when persisting and retrieving sub-images. The LSST convention is that Wcs returns pixel position
     *  (which is based on position in the parent image), but the FITS convention is to return pixel index
     *  (which is bases on position in the sub-image). In order that the FITS files we create make sense
     *  to other FITS viewers, we change to the FITS convention when writing out images.
     */
    virtual void shiftReferencePixel(double dx, double dy);

    // Virtual to make sure subclasses use the correct version of both shiftReferencePixel mutators.
    virtual void shiftReferencePixel(geom::Extent2D const & d) { shiftReferencePixel(d.getX(), d.getY()); }

    /// Whether the Wcs is persistable using afw::table::io archives.
    virtual bool isPersistable() const;

private:
    //Allow the formatter to access private goo
    LSST_PERSIST_FORMATTER(lsst::afw::formatters::WcsFormatter)

    /** Manually initialise a wcs struct using values passed by the constructor
     *
     * @param crval The sky position of the reference point
     * @param crpix The pixel position corresponding to crval in LSST units
     * @param CD Matrix describing transformations from pixel to sky positions
     * @param ctype1 Projection system used (see description of Wcs)
     * @param ctype2 Projection system used (see description of Wcs)
     * @param equinox Equinox of coordinate system, eg 2000 (Julian) or 1950 (Besselian)
     * @param raDecSys System used to describe right ascension or declination, e.g FK4, FK5 or ICRS
     * @param cunits1 Units of sky position. One of deg, arcmin or arcsec
     * @param cunits2 Units of sky position. One of deg, arcmin or arcsec
     */
    void initWcsLib(geom::Point2D const & crval, geom::Point2D const & crpix,
                    Eigen::Matrix2d const & CD,
                    std::string const & ctype1, std::string const & ctype2,
                    double equinox, std::string const & raDecSys,
                    std::string const & cunits1, std::string const & cunits2
                   );

protected:

    friend class WcsFactory;
    /// Perform basic checks on whether *this might be persistable
    bool _mayBePersistable() const;
    // See afw::table::io::Persistable
    virtual std::string getPersistenceName() const;
    virtual std::string getPythonModule() const;
    virtual void write(OutputArchiveHandle & handle) const;

    // Protected virtual implementation for operator== (must be true in both directions for equality).
    virtual bool _isSubset(Wcs const & other) const;

    // Return true if coordinate system is ICRS or FK5 J2000
    bool _isIcrs() const {
        return (getCoordSystem() == afw::coord::ICRS) ||
            ((getCoordSystem() == afw::coord::FK5) && (getEquinox() == 2000));
    }

    // Default constructor, only used by WcsFormatter
    /// Construct an invalid Wcs given no arguments
    Wcs();

    //If you want to create a Wcs from a FITS header, use makeWcs().
    //This is protected because the derived classes need to be able to see it.
    /** Create a Wcs from a fits header.
     *
     * Don't call this directly. Use makeWcs() instead, which will figure out which (if any) sub-class of Wcs is appropriate.
     */
    Wcs(CONST_PTR(lsst::daf::base::PropertySet) const& fitsMetadata);

    // Construct from a record; used by WcsFactory for afw::table::io persistence.
    explicit Wcs(afw::table::BaseRecord const & record);

    ///Copy constructor
    Wcs(Wcs const & rhs);
    Wcs& operator= (const Wcs &);

    /**
     * Worker routine for pixelToSky
     */
    virtual void pixelToSkyImpl(double pixel1, double pixel2, geom::Angle skyTmp[2]) const;
    /**
     * Worker routine for skyToPixel
     *
     * @param sky1 RA (or, more generally, longitude)
     * @param sky2 Dec (or latitude)
     */
    virtual geom::Point2D skyToPixelImpl(geom::Angle sky1, geom::Angle sky2) const;

    /**
     * Given a sky position, use the values stored in ctype and radesys to return the correct sub-class of Coord.
     */
    PTR(afw::coord::Coord) makeCorrectCoord(geom::Angle sky0, geom::Angle sky1) const;

    /**
     *  Given a Coord (as a shared pointer), return the sky position in the correct
     *  coordinate system for this Wcs.
     */
    PTR(afw::coord::Coord) convertCoordToSky(coord::Coord const & coord) const;

    /**
     * Implementation for the overloaded public linearizePixelToSky methods, requiring both a pixel coordinate and the corresponding sky coordinate.
     */
    virtual geom::AffineTransform linearizePixelToSkyInternal(
        geom::Point2D const & pix,
        coord::Coord const & coord,
        geom::AngleUnit skyUnit
    ) const;

    /**
     * Implementation for the overloaded public linearizeSkyToPixel methods, requiring both a pixel coordinate and the corresponding sky coordinate.
     */
    virtual geom::AffineTransform linearizeSkyToPixelInternal(
        geom::Point2D const & pix,
        coord::Coord const & coord,
        geom::AngleUnit skyUnit
    ) const;


    ///Parse a fits header, extract the relevant metadata and create a Wcs object
    void initWcsLibFromFits(CONST_PTR(lsst::daf::base::PropertySet) const& fitsMetadata);
    /**
     * Set some internal variables that we need to refer to
     */
    void _initWcs();
    void _setWcslibParams();

    struct wcsprm* _wcsInfo;
    int _nWcsInfo;
    int _relax; ///< Degree of permissiveness for wcspih (0 for strict); see wcshdr.h for details.
    int _wcsfixCtrl; ///< Do potentially unsafe translations of non-standard unit strings? 0/1 = no/yes
    int _wcshdrCtrl; ///< Controls messages to stderr from wcshdr (0 for none); see wcshdr.h for details
    int _nReject;
    coord::CoordSystem _coordSystem;
    bool _skyAxesSwapped; ///< if true then the sky axes are swapped
};

namespace detail {
    PTR(lsst::daf::base::PropertyList)
    createTrivialWcsAsPropertySet(std::string const& wcsName, int const x0=0, int const y0=0);

    geom::Point2I getImageXY0FromMetadata(std::string const& wcsName, lsst::daf::base::PropertySet *metadata);
}

/**
 * Create a Wcs object from a fits header.
 * It examines the header and determines the
 * most suitable object to return, either a general Wcs object, or a more specific object specialised to a
 * given coordinate system (e.g TanWcs)
 *
 * @param fitsMetadata input metadata
 * @param stripMetadata Remove FITS keywords from metadata?
 */
PTR(Wcs) makeWcs(PTR(lsst::daf::base::PropertySet) const& fitsMetadata, bool stripMetadata=false);

/**
 * Create a Wcs object from crval, crpix, CD, using CD elements (useful from python)
 *
 * @param crval CRVAL1,2 (ie. the sky origin)
 * @param crpix CRPIX1,2 (ie. the pixel origin) in pixels
 * @param CD11 CD matrix element 1,1
 * @param CD12 CD matrix element 1,2
 * @param CD21 CD matrix element 2,1
 * @param CD22 CD matrix element 2,2
 *
 * @note CD matrix elements must be in degrees/pixel.
 */
PTR(Wcs) makeWcs(coord::Coord const & crval, geom::Point2D const & crpix,
                 double CD11, double CD12, double CD21, double CD22);

namespace detail {
    int stripWcsKeywords(PTR(lsst::daf::base::PropertySet) const& metadata, ///< Metadata to be stripped
                         CONST_PTR(Wcs) const& wcs ///< A Wcs with (implied) keywords
                        );
}


/**
 * XYTransformFromWcsPair: An XYTransform obtained by putting two Wcs objects "back to back".
 *
 * Eventually there will be an XYTransform subclass which represents a camera distortion.
 * For now we can get a SIP camera distortion in a clunky way, by using an XYTransformFromWcsPair
 * with a SIP-distorted TanWcs and an undistorted Wcs.
 */
class XYTransformFromWcsPair : public afw::geom::XYTransform
{
public:
    XYTransformFromWcsPair(CONST_PTR(Wcs) dst, CONST_PTR(Wcs) src);
    virtual ~XYTransformFromWcsPair() { }

    virtual PTR(afw::geom::XYTransform) invert() const;

    /// The following methods are needed to devirtualize the XYTransform parent class
    virtual PTR(afw::geom::XYTransform) clone() const;
    virtual Point2D forwardTransform(Point2D const &pixel) const;
    virtual Point2D reverseTransform(Point2D const &pixel) const;

protected:
    CONST_PTR(Wcs) _dst;
    CONST_PTR(Wcs) _src;
    bool const _isSameSkySystem;
};


}}} // lsst::afw::image

#endif // LSST_AFW_IMAGE_WCS_H

//  LocalWords:  LSST
