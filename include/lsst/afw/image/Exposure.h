// -*- LSST-C++ -*- // fixed format comment for emacs
/**
  * \file Exposure.h
  *
  * \class lsst::afw::Exposure 
  *
  * \ingroup fw
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

#include <lsst/daf/data/LsstBase.h>
#include <lsst/pex/persistence/Persistable.h>
#include <lsst/afw/image/MaskedImage.h>
#include <lsst/afw/image/WCS.h>

namespace lsst {
namespace fw {
    
    template<class ImageT, class MaskT> class Exposure;
    namespace formatters {
        template<class ImageT, class MaskT> class ExposureFormatter;
    }
        
    template<typename ImageT, typename MaskT> 
    class Exposure : public lsst::pex::persistence::Persistable,
                     public lsst::daf::data::LsstBase {
                
    public:    

        typedef boost::shared_ptr<lsst::afw::math::WCS> wscPtrType;

        // Class Constructors and Destructor
        explicit Exposure();
        explicit Exposure(MaskedImage<ImageT, MaskT> const &maskedImage);
        explicit Exposure(MaskedImage<ImageT, MaskT> const &maskedImage, WCS const &wcs);
        explicit Exposure(unsigned cols, unsigned rows, WCS const &wcs);
        explicit Exposure(unsigned cols, unsigned rows);
        virtual ~Exposure(); 
        
        // Get Members (getMaskedImage is inline) 
        MaskedImage<ImageT, MaskT> getMaskedImage() const { return _maskedImage; };
        WCS getWcs() const;
        Exposure<ImageT, MaskT> getSubExposure(const vw::BBox2i&) const;
        
        // Set Members
        void setMaskedImage(MaskedImage<ImageT, MaskT> &maskedImage);
        void setWcs(WCS const &wcs);
        
        // Has Member (inline)
        bool hasWcs() const { return static_cast<bool>(_wcsPtr); };
        
        // Read Fits and Write Fits Members
        void readFits(std::string const &expInFile);
        void writeFits(std::string const &expOutFile) const;
        
    private:
        LSST_PERSIST_FORMATTER(formatters::ExposureFormatter<ImageT, MaskT>);

        MaskedImage<ImageT, MaskT> _maskedImage;             
        boost::shared_ptr<WCS> _wcsPtr;    
    };     
}} // fw::lsst

#endif // LSST_AFW_IMAGE_EXPOSURE_H
