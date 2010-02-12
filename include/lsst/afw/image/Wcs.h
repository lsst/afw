// -*- LSST-C++ -*-

#ifndef LSST_AFW_IMAGE_WCS_H
#define LSST_AFW_IMAGE_WCS_H


#include "Eigen/Core.h"
#include "lsst/daf/base.h"
#include "lsst/daf/data/LsstBase.h"
#include "lsst/afw/image/Image.h"

struct wcsprm;                          // defined in wcs.h

namespace lsst {
namespace afw {
    namespace formatters {
        class WcsFormatter;
    }
namespace image {


    class Wcs : public lsst::daf::base::Persistable,
                    public lsst::daf::data::LsstBase {
    public:
        typedef boost::shared_ptr<lsst::afw::image::Wcs> Ptr;
        typedef boost::shared_ptr<lsst::afw::image::Wcs const> ConstPtr;
        
        //Constructors
        Wcs(lsst::daf::base::PropertySet::Ptr fitsMetadata);
        Wcs(lsst::afw::image::PointD crval, lsst::afw::image::PointD crpix, Eigen::Matrix2d CD, 
                const std::string ctype1="RA--TAN", const std::string ctype2="DEC-TAN",
                double equinox=2000, std::string raDecSys="FK5",
                const std::string cunits1="deg", const std::string cunits2="deg"
           );

        virtual ~Wcs();

        //Accessors
        PointD getSkyOrigin() const;      //Return crval
        PointD getPixelOrigin() const;    //Return crpix
        Eigen::Matrix2d getCDMatrix() const;       //Return CD matrix
        
        lsst::daf::base::PropertySet::Ptr getFitsMetadata() const;
        bool isFlipped() const; //Does the Wcs follow the convention of North=Up, East=Left or not
        double pixArea(PointD pix00) const;

        //Convert from raDec to pixel space. Formerly called raDecToXY() and
        //xyToRaDec(), but the name now reflects their increased generality. They may be
        //used, e.g. to convert xy to Galactic coordinates
        virtual PointD skyToPixel(const PointD sky) const;
        virtual PointD pixelToSky(const PointD pixel) const;

        virtual PointD skyToPixel(double sky1, double sky2) const;
        virtual PointD pixelToSky(double pixel1, double pixel2) const;

        //Mutators
        void shiftReferencePixel(double dx, double dy); 

    private:
        void initWcsLib(lsst::afw::image::PointD crval, lsst::afw::image::PointD crpix, Eigen::Matrix2d CD, 
                        const std::string ctype1, const std::string ctype2,
                        double equinox, std::string raDecSys,
                        const std::string cunits1, const std::string cunits2
                       );

        void initWcsLibFromFits(lsst::daf::base::PropertySet::Ptr const fitsMetadata);
    
    protected:

        struct wcsprm* _wcsInfo;
        int _nWcsInfo;
        int _relax; ///< Degree of permissiveness for wcspih (0 for strict); see wcshdr.h for details.
        int _wcsfixCtrl; ///< Do potentially unsafe translations of non-standard unit strings? 0/1 = no/yes
        int _wcshdrCtrl; ///< Controls messages to stderr from wcshdr (0 for none); see wcshdr.h for details
        int _nReject;

    };
}}} // lsst::afw::image

#endif // LSST_AFW_IMAGE_WCS_H
