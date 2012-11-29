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


#include "Eigen/Core"
#include "lsst/base.h"
#include "lsst/daf/base/Persistable.h"
#include "lsst/daf/base/Citizen.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/table/generators.h"

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
namespace fits {
class Fits;
class MemFileManager;
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
/// In its simplest sense, Wcs is used to convert from position in the sky (in
/// right ascension and declination) to pixel position on an image (and back
/// again). It is, however, much more general than that and can understand a
/// myriad of different coordinate systems.
/// 
/// A wcs can be constructed from a reference position (crval, crpix) and a
/// translation matrix. Alternatively, if you have the header from a fits file,
/// you can create a Wcs object with the makeWcs() function. This function
/// determines whether your Wcs is one the subset of projection systems that is
/// dealt with specially by Lsst, and creates an object of the correct
/// class. Otherwise, a pointer to a Wcs object is returned.  Most astronomical
/// images use tangent plane projection, so makeWcs() returns a TanWcs object
/// pointer
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
/// Note that we violate the Wcs standard in one minor way. The standard states
/// that none of the CRPIX or CRVAL keywords are required, for the header to be
/// valid, and the appropriate values should be set to 0.0 if the keywords are
/// absent. This is a recipe for painful bugs in analysis, so we violate the
/// standard by insisting that the keywords CRPIX[1,2] and CRVAL[1,2] are
/// present when reading a header (keywords CRPIX1a etc are also accepted)

class Wcs : public lsst::daf::base::Persistable,
            public lsst::daf::base::Citizen
{
public:
    typedef boost::shared_ptr<lsst::afw::image::Wcs> Ptr;
    typedef boost::shared_ptr<lsst::afw::image::Wcs const> ConstPtr;

    //Create a Wcs of the correct class using a fits header.
    friend Wcs::Ptr makeWcs(PTR(lsst::daf::base::PropertySet) const& fitsMetadata,
                            bool stripMetadata);

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

    // Returns CRVAL
    lsst::afw::coord::Coord::Ptr getSkyOrigin() const;
    // Returns CRPIX
    lsst::afw::geom::Point2D getPixelOrigin() const;
    // Returns CD matrix.  You would never have guessed that from the name.
    Eigen::Matrix2d getCDMatrix() const;
    virtual void flipImage(int flipLR, int flipTB, lsst::afw::geom::Extent2I dimensions) const;
    virtual void rotateImageBy90(int nQuarter, lsst::afw::geom::Extent2I dimensions) const;
    
    virtual PTR(lsst::daf::base::PropertyList) getFitsMetadata() const;
    
    // Does the Wcs follow the convention of North=Up, East=Left or not
    // [This actually just measures the sign of the determinant of the CD matrix
    //  to determine the "handedness" of the coordinate system.]
    bool isFlipped() const;

    // Sky area covered by a pixel at position \c pix00 in units of square degrees.
    double pixArea(lsst::afw::geom::Point2D pix00) const;
    
    // Returns the pixel scale [Angle/pixel]
    lsst::afw::geom::Angle pixelScale() const;
    
    // Convert from raDec to pixel space. Formerly called raDecToXY() and
    // xyToRaDec(), but the name now reflects their increased generality. They
    // may be used, e.g. to convert xy to Galactic coordinates
    lsst::afw::coord::Coord::Ptr pixelToSky(double pix1, double pix2) const;
    lsst::afw::coord::Coord::Ptr pixelToSky(lsst::afw::geom::Point2D const & pixel) const;

    /// \note This routine is designed for the knowledgeable user in need of
    /// performance; it's safer to call the version that returns a CoordPtr
    void pixelToSky(
        double pixel1, double pixel2, lsst::afw::geom::Angle& sky1, lsst::afw::geom::Angle& sky2
    ) const;
    
    // ASSUMES the angles are in the appropriate coordinate system for this WCS.
    lsst::afw::geom::Point2D skyToPixel(lsst::afw::geom::Angle sky1, lsst::afw::geom::Angle sky2) const;

    /// @brief Convert from sky coordinates (e.g ra/dec) to pixel positions.
    lsst::afw::geom::Point2D skyToPixel(lsst::afw::coord::Coord const & coord) const;

    /**
     *  @brief Convert from sky coordinates (e.g ra/dec) to intermediate world coordinates
     *
     *  Intermediate world coordinates are in DEGREES.
     */
    lsst::afw::geom::Point2D skyToIntermediateWorldCoord(lsst::afw::coord::Coord const & coord) const;
    
    virtual bool hasDistortion() const {    return false;};
    
    lsst::afw::geom::LinearTransform getLinearTransform() const;
    
    /**
     * @brief Return the local linear approximation to Wcs::pixelToSky at a point given in sky coordinates.
     *
     * The local linear approximation is defined such the following is true (ignoring floating-point errors):
     * @code
     * wcs.linearizePixelToSky(sky, skyUnit)(wcs.skyToPixel(sky)) == sky.getPosition(skyUnit);
     * @endcode
     * (recall that AffineTransform::operator() is matrix multiplication with the augmented point (x,y,1)).
     *
     * This is currently implemented as a numerical derivative, but we should specialise the Wcs class
     * (or rather its implementation) to handle "simple" cases such as TAN-SIP analytically
     *
     * @param[in] coord   Position in sky coordinates where transform is desired.
     * @param[in] skyUnit Units to use for sky coordinates; units of matrix elements will be skyUnits/pixel.
     */
    lsst::afw::geom::AffineTransform linearizePixelToSky(
        lsst::afw::coord::Coord const & coord,
        lsst::afw::geom::AngleUnit skyUnit = lsst::afw::geom::degrees
    ) const;
    
    /**
     * @brief Return the local linear approximation to Wcs::pixelToSky at a point given in pixel coordinates.
     *
     * The local linear approximation is defined such the following is true (ignoring floating-point errors):
     * @code
     * wcs.linearizePixelToSky(pix, skyUnit)(pix) == wcs.pixelToSky(pix).getPosition(skyUnit)
     * @endcode
     * (recall that AffineTransform::operator() is matrix multiplication with the augmented point (x,y,1)).
     *
     * This is currently implemented as a numerical derivative, but we should specialise the Wcs class
     * (or rather its implementation) to handle "simple" cases such as TAN-SIP analytically
     *
     * @param[in] pix     Position in pixel coordinates where transform is desired.
     * @param[in] skyUnit Units to use for sky coordinates; units of matrix elements will be skyUnits/pixel.
     */
    lsst::afw::geom::AffineTransform linearizePixelToSky(
        lsst::afw::geom::Point2D const & pix,
        lsst::afw::geom::AngleUnit skyUnit = lsst::afw::geom::degrees
    ) const;

    /**
     * @brief Return the local linear approximation to Wcs::skyToPixel at a point given in sky coordinates.
     *
     * The local linear approximation is defined such the following is true (ignoring floating-point errors):
     * @code
     * wcs.linearizeSkyToPixel(sky, skyUnit)(sky.getPosition(skyUnit)) == wcs.skyToPixel(sky)
     * @endcode
     * (recall that AffineTransform::operator() is matrix multiplication with the augmented point (x,y,1)).
     *
     * This is currently implemented as a numerical derivative, but we should specialise the Wcs class
     * (or rather its implementation) to handle "simple" cases such as TAN-SIP analytically
     *
     * @param[in] coord   Position in sky coordinates where transform is desired.
     * @param[in] skyUnit Units to use for sky coordinates; units of matrix elements will be pixels/skyUnit.
     */
    lsst::afw::geom::AffineTransform linearizeSkyToPixel(
        lsst::afw::coord::Coord const & coord,
        lsst::afw::geom::AngleUnit skyUnit = lsst::afw::geom::degrees
    ) const;

    /**
     * @brief Return the local linear approximation to Wcs::skyToPixel at a point given in pixel coordinates.
     *
     * The local linear approximation is defined such the following is true (ignoring floating-point errors):
     * @code
     * wcs.linearizeSkyToPixel(pix, skyUnit)(wcs.pixelToSky(pix).getPosition(skyUnit)) == pix
     * @endcode
     * (recall that AffineTransform::operator() is matrix multiplication with the augmented point (x,y,1)).
     *
     * This is currently implemented as a numerical derivative, but we should specialise the Wcs class
     * (or rather its implementation) to handle "simple" cases such as TAN-SIP analytically
     *
     * @param[in] pix     Position in pixel coordinates where transform is desired.
     * @param[in] skyUnit Units to use for sky coordinates; units of matrix elements will be pixels/skyUnit.
     */        
    lsst::afw::geom::AffineTransform linearizeSkyToPixel(
        lsst::afw::geom::Point2D const & pix,
        lsst::afw::geom::AngleUnit skyUnit = lsst::afw::geom::degrees
    ) const;

    //Mutators
    void shiftReferencePixel(geom::Extent2D const & d) {shiftReferencePixel(d.getX(), d.getY());}
    void shiftReferencePixel(double dx, double dy);

    /**
     *  @name Table-based Persistence
     *
     *  While the FITS standard for WCS representation is header-based, we also provide custom persistence
     *  to one or more FITS binary tables for two reasons:
     *   - It allows us to persist complex, custom astrometric models that do not have a FITS standard
     *     representation.
     *   - It allows us to save many Wcs objects in a single FITS file more efficiently, which allows
     *     us to save the coordinate systems of all input exposures in a coadd FITS file.
     *
     *  @sa Exposure::writeFits
     */
    ///@{

    /// @brief Return true if the writeToRecords() and readFromRecords() methods are supported for this Wcs.
    virtual bool hasRecordPersistence() const;

#ifndef SWIG // only expose high-level interface to Python

    /**
     *  @brief Return objects that allow the Wcs to be written to one or more RecordOutputGenerators.
     *
     *  The resulting records can be fed into one more RecordInputGenerators to create an equivalent
     *  Wcs by using readFromRecords().
     *
     *  The default implementation throws an exception.
     *
     *  The user is responsible for ensuring that the lifetime of the returned object does not
     *  exceed the lifetime of the Wcs, as it may contain a non-owning back reference to the Wcs.
     */
    virtual afw::table::RecordOutputGeneratorSet writeToRecords() const;

    /**
     *  @brief Create a Wcs from one or more RecordInputGenerators.
     *
     *  The records should have been created by writeToRecords.  This is implemented by
     *  WcsRecordGeneratorFactory.
     */
    static PTR(Wcs) readFromRecords(afw::table::RecordInputGeneratorSet const & inputs);

    /**
     *  @brief Write the Wcs to an already-open FITS object.
     *
     *  @param[in] fitsfile     Open FITS object to write to.
     */
    void writeFitsTables(fits::Fits & fitsfile) const;

    /**
     *  @brief Read a Wcs from an already open FITS object.
     *
     *  @param[in]  fitsfile     FITS object to read from, already positioned at the desired HDU.
     */
    static PTR(Wcs) readFitsTables(fits::Fits & fitsfile);

#endif // !SWIG

    /**
     *  @brief Write the Wcs to a regular FITS file.
     *
     *  @param[in] fileName     Name of the file to write to.
     *  @param[in] mode         If "w", any existing file with the given name will be overwritten.  If
     *                          "a", new HDUs will be appended to an existing file.
     */
    void writeFitsTables(std::string const & fileName, std::string const & mode="w") const;

    /**
     *  @brief Write the Wcs to a FITS image in memory.
     *
     *  @param[in] manager      Name of the file to write to.
     *  @param[in] mode         If "w", any existing file with the given name will be overwritten.  If
     *                          "a", new HDUs will be appended to an existing file.
     */
    void writeFitsTables(fits::MemFileManager & manager, std::string const & mode="w") const;

    /**
     *  @brief Read a Wcs from a regular FITS file.
     *
     *  @param[in]  fileName     Name of the file to read.
     *  @param[in]  hdu          HDU to read, where 1 is the primary.  The special value of 0
     *                           skips the primary HDU if it is empty.
     */
    static PTR(Wcs) readFitsTables(std::string const & fileName, int hdu=0);

    /**
     *  @brief Read a Wcs from a FITS file in memory.
     *
     *  @param[in]  manager      Manager for the memory to read from.
     *  @param[in]  hdu          HDU to read, where 1 is the primary.  The special value of 0
     *                           skips the primary HDU if it is empty.
     */
    static PTR(Wcs) readFitsTables(fits::MemFileManager & manager, int hdu=0);

    ///@}
        
private:
    //Allow the formatter to access private goo
    LSST_PERSIST_FORMATTER(lsst::afw::formatters::WcsFormatter)
    
    void initWcsLib(lsst::afw::geom::Point2D const & crval, lsst::afw::geom::Point2D const & crpix,
                    Eigen::Matrix2d const & CD, 
                    std::string const & ctype1, std::string const & ctype2,
                    double equinox, std::string const & raDecSys,
                    std::string const & cunits1, std::string const & cunits2
                   );

    virtual void pixelToSkyImpl(double pixel1, double pixel2, lsst::afw::geom::Angle skyTmp[2]) const;
    virtual lsst::afw::geom::Point2D skyToPixelImpl(lsst::afw::geom::Angle sky1, lsst::afw::geom::Angle sky2) const;

protected:

    // Inner class used for record persistence, defined here so subclasses of Wcs can subclass it.
    friend class WcsRecordOutputGenerator;

    friend class WcsRecordGeneratorFactory;

    // Protected virtual implementation for operator== (must be true in both directions for equality).
    virtual bool _isSubset(Wcs const & other) const;

    // Default constructor, only used by WcsFormatter
    Wcs();

    //If you want to create a Wcs from a fits header, use makeWcs(). 
    //This is protected because the derived classes need to be able to see it.
    Wcs(CONST_PTR(lsst::daf::base::PropertySet) const& fitsMetadata);

    // Constructor used in record persistence, implemented in WcsRecordGeneratorFactory.cc
    Wcs(afw::table::BaseRecord const & record);
    
    Wcs(lsst::afw::image::Wcs const & rhs);
    Wcs& operator= (const Wcs &);        
    
    afw::coord::Coord::Ptr makeCorrectCoord(lsst::afw::geom::Angle sky0, lsst::afw::geom::Angle sky1) const;

    /**
     *  Given a Coord (as a shared pointer), return the sky position in the correct
     *  coordinate system for this Wcs.
     */
    afw::coord::Coord::Ptr convertCoordToSky(lsst::afw::coord::Coord const & coord) const;
    
    virtual lsst::afw::geom::AffineTransform linearizePixelToSkyInternal(
        lsst::afw::geom::Point2D const & pix,
        lsst::afw::coord::Coord const & coord,
        lsst::afw::geom::AngleUnit skyUnit
    ) const;

    virtual lsst::afw::geom::AffineTransform linearizeSkyToPixelInternal(
        lsst::afw::geom::Point2D const & pix,
        lsst::afw::coord::Coord const & coord,
        lsst::afw::geom::AngleUnit skyUnit
    ) const;

    
    void initWcsLibFromFits(CONST_PTR(lsst::daf::base::PropertySet) const& fitsMetadata);
    void _initWcs();
    void _setWcslibParams();
    
    struct wcsprm* _wcsInfo;
    int _nWcsInfo;
    int _relax; ///< Degree of permissiveness for wcspih (0 for strict); see wcshdr.h for details.
    int _wcsfixCtrl; ///< Do potentially unsafe translations of non-standard unit strings? 0/1 = no/yes
    int _wcshdrCtrl; ///< Controls messages to stderr from wcshdr (0 for none); see wcshdr.h for details
    int _nReject;
    lsst::afw::coord::CoordSystem _coordSystem;
};

namespace detail {
    PTR(lsst::daf::base::PropertyList)
    createTrivialWcsAsPropertySet(std::string const& wcsName, int const x0=0, int const y0=0);
    
    geom::Point2I getImageXY0FromMetadata(std::string const& wcsName, lsst::daf::base::PropertySet *metadata);

    int stripWcsKeywords(PTR(lsst::daf::base::PropertySet) const& metadata, ///< Metadata to be stripped
                         CONST_PTR(Wcs) const& wcs ///< A Wcs with (implied) keywords
                        );
} // namespace detail

Wcs::Ptr makeWcs(PTR(lsst::daf::base::PropertySet) const& fitsMetadata, bool stripMetadata=false);

/*
 Note, CD matrix elements must be in degrees/pixel.
 */
Wcs::Ptr makeWcs(lsst::afw::coord::Coord const & crval, lsst::afw::geom::Point2D const & crpix,
                 double CD11, double CD12, double CD21, double CD22);

}}} // lsst::afw::image

#endif // LSST_AFW_IMAGE_WCS_H

//  LocalWords:  LSST
