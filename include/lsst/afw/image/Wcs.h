// -*- lsst-c++ -*-
/**
 * \file
 * \brief Support for Astrometry
 *
 * This is my draft of an improvement to the WCS class. It goes in here because this is more of a scratch
 * directory, and rhl might cringe if he sees this in the real development code
 */

#ifndef LSST_AFW_IMAGE_WCS_H
#define LSST_AFW_IMAGE_WCS_H

#include "boost/numeric/ublas/matrix.hpp"
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
        Wcs(lsst::daf::base::DataProperty::PtrType fitsMetadata);
        Wcs(PointD crval, PointD crpix, boost::numeric::ublas::matrix<double> CD);
        Wcs(Wcs const &);
        Wcs & operator = (const Wcs &);

        ~Wcs();

        /// Return the input fits header
        lsst::daf::base::DataProperty::PtrType getFitsMetadata() const { 
            return _fitsMetadata; 
        }

        /// Return true iff Wcs is valid
        operator bool() const { return _wcsInfo != NULL; }

        lsst::afw::image::PointD getRaDecCenter() const;
        lsst::afw::image::PointD getColRowCenter() const;
        boost::numeric::ublas::matrix<double> getLinearTransformMatrix() const;
        
        lsst::afw::image::PointD raDecToColRow(lsst::afw::image::PointD sky) const;
        lsst::afw::image::PointD raDecToColRow(double const ra, double const dec) const;
        lsst::afw::image::PointD raDecToColRow(double const radec[2]) const {
            return raDecToColRow(radec[0], radec[1]);
        }

        lsst::afw::image::PointD colRowToRaDec(lsst::afw::image::PointD pix) const;
        lsst::afw::image::PointD colRowToRaDec(double const x, double const y) const;
        lsst::afw::image::PointD colRowToRaDec(double const xy[2]) const {
            return colRowToRaDec(xy[0], xy[1]);
        }

        double pixArea(lsst::afw::image::PointD pix) const;
    private:
        LSST_PERSIST_FORMATTER(lsst::afw::formatters::WcsFormatter);

        lsst::daf::base::DataProperty::PtrType _fitsMetadata; ///< Input FITS header.  Caveat Emptor: may contain other keywords
        // including e.g. SIMPLE and BITPIX
        struct wcsprm* _wcsInfo;
        int _nWcsInfo;
        int _relax; ///< Degree of permissiveness for wcspih (0 for strict); see wcshdr.h for details.
        int _wcsfixCtrl; ///< Do potentially unsafe translations of non-standard unit strings? 0/1 = no/yes
        int _wcshdrCtrl; ///< Controls messages to stderr from wcshdr (0 for none); see wcshdr.h for details
        int _nReject;
    };
  
}}} // lsst::afw::image

#endif // LSST_AFW_IMAGE_WCS_H
