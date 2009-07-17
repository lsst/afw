// -*- lsst-c++ -*-
/**
 * \file
 * \brief Support for Astrometry
 *
 */

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
    /// \brief Wcs supports coordinate system transformations between pixel and world coordinates
    ///
    /// All Wcs (in the FITS sense) coordinate conventions are supported via
    /// Mark Calabretta's wcslib package (http://www.atnf.csiro.au/people/mcalabre)
    ///
    class Wcs : public lsst::daf::base::Persistable,
                public lsst::daf::data::LsstBase {
    public:
        typedef boost::shared_ptr<lsst::afw::image::Wcs> Ptr;
        
        Wcs();
        Wcs(lsst::daf::base::PropertySet::Ptr fitsMetadata);
        Wcs(PointD crval, PointD crpix, Eigen::Matrix2d CD, double equinox=2000.0,
            std::string raDecSys="FK5");
        Wcs(PointD crval, PointD crpix, Eigen::Matrix2d CD, 
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> sipA, ///< Forward distortion Matrix A
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> sipB, ///< Forward distortion Matrix B
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> sipAp, ///<Reverse distortion Matrix Ap
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> sipBp,  ///<Reverse distortion Matrix Bp
            double equinox=2000.0,
            std::string raDecSys="FK5"
           );

        Wcs(Wcs const &);
        Wcs & operator = (const Wcs &);

        ~Wcs();

        lsst::daf::base::PropertySet::Ptr getFitsMetadata() const;

        /// Return true iff Wcs is valid
        operator bool() const { return _wcsInfo != NULL; }

        bool isFlipped();

        void shiftReferencePixel(double const dx, double const dy);

        lsst::afw::image::PointD getOriginRaDec() const;
        lsst::afw::image::PointD getOriginXY() const;
            
        PointD raDecToXY(PointD sky) const;
        PointD raDecToXY(double const ra, double const dec) const;
        PointD raDecToXY(double const radec[2]) const {
            return raDecToXY(radec[0], radec[1]);
        }
        Eigen::Matrix<double, 2, 2> getLinearTransformMatrix() const;

        PointD xyToRaDec(PointD pix) const;
        PointD xyToRaDec(double const x, double const y) const;
        PointD xyToRaDec(double const xy[2]) const {
            return xyToRaDec(xy[0], xy[1]);
        }

        double pixArea(lsst::afw::image::PointD pix) const;


    private:
        void initWcslib(PointD crval,                   
                        PointD crpix,                   
                        Eigen::Matrix2d CD,             
                        double equinox,                 
                        std::string raDecSys,           
                        std::string ctype1="RA---TAN",  
                        std::string ctype2="DEC--TAN"   
                        );
        void setCtypesToLinear();
        void setCtypesToSIP();




        
        LSST_PERSIST_FORMATTER(lsst::afw::formatters::WcsFormatter);

        struct wcsprm* _wcsInfo;
        int _nWcsInfo;
        int _relax; ///< Degree of permissiveness for wcspih (0 for strict); see wcshdr.h for details.
        int _wcsfixCtrl; ///< Do potentially unsafe translations of non-standard unit strings? 0/1 = no/yes
        int _wcshdrCtrl; ///< Controls messages to stderr from wcshdr (0 for none); see wcshdr.h for details
        int _nReject;

        //SIP keywords
        Eigen::MatrixXd _sipA, _sipB; ///< Forward transformation
        Eigen::MatrixXd _sipAp, _sipBp;   ///<Reverse transformation
        
    };
  
    namespace detail {
        lsst::daf::base::PropertySet::Ptr
        createTrivialWcsAsPropertySet(std::string const& wcsName, int const x0=0, int const y0=0);

        image::PointI getImageXY0FromMetadata(std::string const& wcsName, lsst::daf::base::PropertySet *metadata);
    }

}}} // lsst::afw::image

#endif // LSST_AFW_IMAGE_WCS_H
