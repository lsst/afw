// -*- LSST-C++ -*- // fixed format comment for emacs
/**
  * \file Exposure.h
  *
  * \class lsst::afw::image::Exposure 
  *
  * \ingroup afw
  *
  * \brief Declaration of the templated Exposure Class for LSST.  Create an
  * Exposure from a lsst::afw::image::MaskedImage.
  *
  * \author Nicole M. Silvestri, University of Washington
  *
  * Contact: nms@astro.washington.edu
  *
  * Created on: Mon Apr 23 1:01:14 2007
  *
  * \version 
  *
  * LSST Legalese here...
  */

#ifndef LSST_AFW_IMAGE_EXPOSURE_H
#define LSST_AFW_IMAGE_EXPOSURE_H

#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>

#include <vw/Math/BBox.h>

#include <lsst/daf/base/Persistable.h>
#include <lsst/daf/data/LsstBase.h>
#include <lsst/afw/image/MaskedImage.h>
#include <lsst/afw/image/Wcs.h>

namespace lsst {
namespace afw {
    namespace formatters {
        template<class ImageT, class MaskT> class ExposureFormatter;
    }
namespace image {
    
    template<class ImageT, class MaskT> class Exposure;
        
    template<typename ImageT, typename MaskT> 
    class Exposure : public lsst::daf::base::Persistable,
                     public lsst::daf::data::LsstBase {
                
    public:    

        typedef boost::shared_ptr<lsst::afw::image::Wcs> wscPtrType;

        // Class Constructors and Destructor
        explicit Exposure();
        explicit Exposure(MaskedImage<ImageT, MaskT> const &maskedImage);
        explicit Exposure(MaskedImage<ImageT, MaskT> const &maskedImage, Wcs const &wcs);
        explicit Exposure(unsigned cols, unsigned rows, Wcs const &wcs);
        explicit Exposure(unsigned cols, unsigned rows);
        virtual ~Exposure(); 
        
        // Get Members (getMaskedImage is inline) 
        MaskedImage<ImageT, MaskT> getMaskedImage() const { return _maskedImage; };
        Wcs getWcs() const;
        Exposure<ImageT, MaskT> getSubExposure(const vw::BBox2i&) const;
        
        // Set Members
        void setMaskedImage(MaskedImage<ImageT, MaskT> &maskedImage);
        void setWcs(Wcs const &wcs);
        
        // Has Member (inline)
        bool hasWcs() const { return static_cast<bool>(_wcsPtr); };
        
        // Read Fits and Write Fits Members
        void readFits(std::string const &expInFile);
        void writeFits(std::string const &expOutFile) const;
        
    private:
        LSST_PERSIST_FORMATTER(lsst::afw::formatters::ExposureFormatter<ImageT, MaskT>);

        MaskedImage<ImageT, MaskT> _maskedImage;             
        boost::shared_ptr<Wcs> _wcsPtr;    
    };     
}}} // lsst::afw::image

#endif // LSST_AFW_IMAGE_EXPOSURE_H
