// -*- LSST-C++ -*-

#ifndef LSST_AFW_IMAGE_WCS_H
#define LSST_AFW_IMAGE_WCS_H


#include "Eigen/Core.h"
#include "lsst/daf/base.h"
#include "lsst/daf/data/LsstBase.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/deprecated.h"

struct wcsprm;                          // defined in wcs.h

namespace lsst {
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
///  * Greisen & Calabretta, 2002 A&A 395:1061
///  * Calabretta & Greisen, 2002, A&A 395, 1077
///  
/// In it's simplest sense, Wcs is used to convert from position in the sky (in right ascension 
/// and declination) to pixel position on an image (and back again). It is, however, much more general 
/// than that and can understand a myriad of different coordinate systems.
/// 
/// \code
/// import lsst.afw.image as afwImg
/// fitsHeader = afwImg.readMetadata(filename)
/// 
/// if 0:
///     #This doesn't work. See below
///     wcs = afwImg.Wcs(fitsHeader)
///     
/// wcs = afwImg.makeWcs(fitsHeader)
/// 
/// pixelPosition = wcs.skyToPixel(ra, dec)
/// skyPosition = wcs.skyToPixel(xPosition, yPosition)
/// \endcode
/// 
/// A wcs can be constructed from a reference position (crval, crpix) and a translation matrix. Alternatively,
/// if you have the header from a fits file, you can create a Wcs object with the makeWcs() function. This
/// function determines whether your Wcs is one the subset of projection systems that is dealt with specially
/// by Lsst, and creates an object of the correct class.
/// 
/// 
/// This class is implemented in by calls to the wcslib library
/// by Mark Calabretta http://www.atnf.csiro.au/people/mcalabre/WCS/
/// 
    class Wcs : public lsst::daf::base::Persistable,
                    public lsst::daf::data::LsstBase {
    public:
        typedef boost::shared_ptr<lsst::afw::image::Wcs> Ptr;
        typedef boost::shared_ptr<lsst::afw::image::Wcs const> ConstPtr;
        
        //Constructors
        Wcs();
        //Create a Wcs of the correct class using a fits header.
        friend Ptr lsst::afw::image::makeWcs(lsst::daf::base::PropertySet::Ptr fitsMetadata);
        Wcs(const lsst::afw::image::PointD crval, const lsst::afw::image::PointD crpix, const Eigen::Matrix2d &CD, 
                const std::string ctype1="RA---TAN", const std::string ctype2="DEC--TAN",
                double equinox=2000, std::string raDecSys="FK5",
                const std::string cunits1="deg", const std::string cunits2="deg"
           );

        Wcs(lsst::afw::image::Wcs const & rhs);
        Wcs & operator = (const Wcs &);        
        virtual ~Wcs();

        //Accessors
        PointD getSkyOrigin() const;      //Return crval
        PointD getPixelOrigin() const;    //Return crpix
        Eigen::Matrix2d getCDMatrix() const;       //Return CD matrix
        
        virtual lsst::daf::base::PropertySet::Ptr getFitsMetadata() const;

        /// Return true iff Wcs is valid
        operator bool() const { return _nWcsInfo != 0; }

        bool isFlipped() const; //Does the Wcs follow the convention of North=Up, East=Left or not
        double pixArea(PointD pix00) const;

        //Convert from raDec to pixel space. Formerly called raDecToXY() and
        //xyToRaDec(), but the name now reflects their increased generality. They may be
        //used, e.g. to convert xy to Galactic coordinates
        virtual PointD skyToPixel(const PointD sky) const;
        virtual PointD pixelToSky(const PointD pixel) const;

        virtual PointD skyToPixel(double sky1, double sky2) const;
        virtual PointD pixelToSky(double pixel1, double pixel2) const;

        PointD skyRadiansToPixel(double sky1Radians, double sky2Radians) const;
        PointD pixelToSkyRadians(double pixel1, double pixel2) const;

        lsst::afw::geom::AffineTransform getAffineTransform() const;
        lsst::afw::geom::AffineTransform linearizeAt(lsst::afw::image::PointD const & sky) const;

        //Mutators
        void shiftReferencePixel(double dx, double dy); 

        
    private:
        //Allow the formatter to access private goo
        LSST_PERSIST_FORMATTER(lsst::afw::formatters::WcsFormatter);
        
        void initWcsLib(const lsst::afw::image::PointD crval, const lsst::afw::image::PointD crpix, const  Eigen::Matrix2d CD, 
                        const std::string ctype1, const std::string ctype2,
                        double equinox, std::string raDecSys,
                        const std::string cunits1, const std::string cunits2
                       );

        void initWcsLibFromFits(lsst::daf::base::PropertySet::Ptr const fitsMetadata);

    protected:

        //If you want to create a Wcs from a fits header, use makeWcs(). 
        //This is protected because the derived classes need to be able to see it.
        Wcs(lsst::daf::base::PropertySet::Ptr fitsMetadata);

        struct wcsprm* _wcsInfo;
        int _nWcsInfo;
        int _relax; ///< Degree of permissiveness for wcspih (0 for strict); see wcshdr.h for details.
        int _wcsfixCtrl; ///< Do potentially unsafe translations of non-standard unit strings? 0/1 = no/yes
        int _wcshdrCtrl; ///< Controls messages to stderr from wcshdr (0 for none); see wcshdr.h for details
        int _nReject;

    };

    //@TODO: Image.cc doesn't compile without this, although I'm not sure what they do, or if they're
    //necessary
    namespace detail {
        lsst::daf::base::PropertySet::Ptr
        createTrivialWcsAsPropertySet(std::string const& wcsName, int const x0=0, int const y0=0);

        image::PointI getImageXY0FromMetadata(std::string const& wcsName, lsst::daf::base::PropertySet *metadata);
    }


}}} // lsst::afw::image

#endif // LSST_AFW_IMAGE_WCS_H
