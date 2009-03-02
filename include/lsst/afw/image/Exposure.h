// -*- LSST-C++ -*- // fixed format comment for emacs
/**
  * @file
  *
  * @brief Declaration of the templated Exposure Class for LSST.
  *
  * Create an Exposure from a lsst::afw::image::MaskedImage.
  *
  * @ingroup afw
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
        template<typename ImageT, typename MaskT, typename VarianceT> class ExposureFormatter;
    }
namespace image {
    /// A class to contain the data, WCS, and other information needed to describe an %image of the sky
    template<typename ImageT, typename MaskT=lsst::afw::image::MaskPixel,
             typename VarianceT=lsst::afw::image::VariancePixel>
    class Exposure : public lsst::daf::base::Persistable,
                     public lsst::daf::data::LsstBase {
    public:
        typedef MaskedImage<ImageT, MaskT, VarianceT> MaskedImageT;
        typedef boost::shared_ptr<Exposure> Ptr;
        
        // Class Constructors and Destructor
        explicit Exposure(int const cols=0, int const rows=0, Wcs const& wcs=Wcs());
        explicit Exposure(MaskedImageT & maskedImage, Wcs const& wcs=Wcs());
        explicit Exposure(std::string const &baseName, int const hdu=0, BBox const& bbox=BBox(), bool const conformMasks=false);

        Exposure(Exposure const &src, BBox const& bbox, bool const deep=false);

        virtual ~Exposure(); 

        // Get Members
        MaskedImageT& getMaskedImage() { return _maskedImage; };
        MaskedImageT const& getMaskedImage() const { return _maskedImage; };
        Wcs::Ptr getWcs() const;
        
        // Set Members
        void setMaskedImage(MaskedImageT &maskedImage);
        void setWcs(Wcs const& wcs);
        
        // Has Member (inline)
        bool hasWcs() const { return static_cast<bool>(*_wcs); };
        
        // FITS
        void writeFits(std::string const &expOutFile) const;
        
    private:
        LSST_PERSIST_FORMATTER(lsst::afw::formatters::ExposureFormatter<ImageT, MaskT, VarianceT>);

        MaskedImageT _maskedImage;             
        Wcs::Ptr _wcs;
    };

/**
 * A function to return an Exposure of the correct type (cf. std::make_pair)
 */
    template<typename MaskedImageT>
    Exposure<typename MaskedImageT::Image::Pixel>* makeExposure(MaskedImageT & mimage, ///< the Exposure's image
                                                                Wcs const& wcs=Wcs() ///< the Exposure's WCS
                                                               ) {
        return new Exposure<typename MaskedImageT::Image::Pixel>(mimage, wcs);
    }

}}} // lsst::afw::image

#endif // LSST_AFW_IMAGE_EXPOSURE_H
