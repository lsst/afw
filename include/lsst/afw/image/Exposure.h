// -*- LSST-C++ -*- // fixed format comment for emacs
/**
  * @file
  *
  * @class lsst::afw::image::Exposure 
  *
  * @ingroup afw
  *
  * @brief Declaration of the templated Exposure Class for LSST.  Create an
  * Exposure from a lsst::afw::image::MaskedImage.
  *
  * @author Nicole M. Silvestri, University of Washington
  *
  * Contact: nms@astro.washington.edu
  *
  * Created on: Mon Apr 23 1:01:14 2007
  *
  * @version 
  *
  * LSST Legalese here...
  */

#ifndef LSST_AFW_IMAGE_EXPOSURE_H
#define LSST_AFW_IMAGE_EXPOSURE_H

#include "boost/cstdint.hpp"
#include "boost/shared_ptr.hpp"

#include "lsst/daf/base/Persistable.h"
#include "lsst/daf/data/LsstBase.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/image/Wcs.h"

namespace lsst {
namespace afw {
    namespace formatters {
        template<typename ImageT, typename MaskT> class ExposureFormatter;
    }
namespace image {
    template<typename ImageT, typename MaskT=lsst::afw::image::MaskPixel,
             typename VarianceT=lsst::afw::image::VariancePixel>
    class Exposure : public lsst::daf::base::Persistable,
                     public lsst::daf::data::LsstBase {
    public:    
        // Class Constructors and Destructor
        explicit Exposure(int const cols=0, int const rows=0, Wcs const& wcs=Wcs());
        explicit Exposure(MaskedImage<ImageT, MaskT, VarianceT> & maskedImage, Wcs const& wcs=Wcs());
        explicit Exposure(std::string const &baseName, int const hdu=0, bool const conformMasks=false);

        Exposure(Exposure const &src, BBox const& bbox, bool const deep=false);

        virtual ~Exposure(); 

        // Get Members
        MaskedImage<ImageT, MaskT, VarianceT> getMaskedImage() const { return _maskedImage; };
        lsst::daf::base::DataProperty::PtrType getMetaData() const { return _metaData; }
        Wcs::Ptr getWcs() const;
        
        // Set Members
        void setMaskedImage(MaskedImage<ImageT, MaskT, VarianceT> &maskedImage);
        void setWcs(Wcs const& wcs);
        
        // Has Member (inline)
        bool hasWcs() const { return static_cast<bool>(*_wcsPtr); };
        
        // FITS
        void writeFits(std::string const &expOutFile) const;
        
    private:
        LSST_PERSIST_FORMATTER(lsst::afw::formatters::ExposureFormatter<ImageT, MaskT>);

        lsst::daf::base::DataProperty::PtrType _metaData;
        MaskedImage<ImageT, MaskT> _maskedImage;             
        Wcs::Ptr _wcsPtr;
    };     
}}} // lsst::afw::image

#endif // LSST_AFW_IMAGE_EXPOSURE_H
