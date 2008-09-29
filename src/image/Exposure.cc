// -*- LSST-C++ -*- // fixed format comment for emacs
/**
  * @file
  *
  * @ingroup afw
  *
  * @brief Implementation of the Exposure Class for LSST.  Class declaration in
  * Exposure.h.
  *
  * @author Nicole M. Silvestri, University of Washington
  *
  * Contact: nms@astro.washington.edu
  *
  * Created on: Mon Apr 23 13:01:15 2007
  *
  * @version 
  *
  * LSST Legalese here... 
  */

#include <stdexcept>

#include "boost/cstdint.hpp" 
#include "boost/format.hpp" 
#include "boost/shared_ptr.hpp"

#include "lsst/daf/base/DataProperty.h"
#include "lsst/daf/data/LsstBase.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Exposure.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/image/Wcs.h" 
#include "lsst/afw/formatters/WcsFormatter.h"

/** @brief Exposure Class Implementation for LSST: a templated framework class
  * for creating an Exposure from a MaskedImage and a Wcs.
  *
  * An Exposure is required to take one lsst::afw::image::MaskedImage or a region (col,
  * row) defining the size of a MaskedImage (this can be of size 0,0).  An
  * Exposure can (but is not required to) contain a lsst::afw::image::Wcs.
  *
  * The template types should optimally be a float, double, unsigned int 16 bit,
  * or unsigned int 32 bit for the image (pixel) type and an unsigned int 32 bit
  * for the mask type.  These types have been explicitly instantiated for the
  * Exposure class.  All MaskedImage and Wcs constructors are 'const' to allow
  * for views and copying.
  *
  * An Exposure can get and return its MaskedImage, Wcs, and a subExposure.
  * The getSubExposure member takes a BBox region defining the subRegion of 
  * the original Exposure to be returned.  The member retrieves the MaskedImage
  * corresponding to the subRegion.  The MaskedImage class throws an exception
  * for any subRegion extending beyond the original MaskedImage bounding
  * box. This member is not yet fully implemented because it requires the Wcs
  * class to return the Wcs metadata to the member so the CRPIX values of the
  * Wcs can be adjusted to reflect the new subMaskedImage origin.  The
  * getSubExposure member will eventually return a subExposure consisting of   
  * the subMAskedImage and the Wcs object with its corresponding adjusted
  * metadata.
  *
  * The hasWcs member is used to determine if the Exposure has a Wcs.  It is not
  * required to have one.
  */

// CLASS CONSTRUCTORS and DESTRUCTOR

/** @brief Construct a blank Exposure of size 0x0 with no Wcs. 
  */     
template<typename ImageT, typename MaskT, typename VarianceT> 
lsst::afw::image::Exposure<ImageT, MaskT, VarianceT>::Exposure() : 
    lsst::daf::data::LsstBase(typeid(this)),   
    _metaData(lsst::daf::base::DataProperty::createPropertyNode("FitsMetaData")),
    _maskedImage(0,0),
    _wcsPtr()                 
{}


/** @brief Construct an Exposure without a Wcs.
  */       
template<typename ImageT, typename MaskT, typename VarianceT> 
lsst::afw::image::Exposure<ImageT, MaskT, VarianceT>::Exposure( 
    lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> const &maskedImage ///< the MaskedImage
                                                              ) : 
    lsst::daf::data::LsstBase(typeid(this)),   
    _metaData(lsst::daf::base::DataProperty::createPropertyNode("FitsMetaData")),
    _maskedImage(maskedImage),
    _wcsPtr()
{}


/** @brief Construct an Exposure with a Wcs.
  */               
template<typename ImageT, typename MaskT, typename VarianceT> 
lsst::afw::image::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> const &maskedImage, ///< the MaskedImage
    lsst::afw::image::Wcs const &wcs                                 ///< the Wcs
    ) : 
    lsst::daf::data::LsstBase(typeid(this)),
    _metaData(lsst::daf::base::DataProperty::createPropertyNode("FitsMetaData")),
    _maskedImage(maskedImage),
    _wcsPtr(new lsst::afw::image::Wcs (wcs))
{}


/** @brief Construct an Exposure with a blank MaskedImage of specified size and
  * a Wcs.
  */          
template<typename ImageT, typename MaskT, typename VarianceT> 
lsst::afw::image::Exposure<ImageT, MaskT, VarianceT>::Exposure(int cols, ///< number of columns in the MaskedImage
                                                               int rows, ///< number of rows in the MaskedImage
                                                               lsst::afw::image::Wcs const &wcs  ///< the Wcs
                                                              ) :
    lsst::daf::data::LsstBase(typeid(this)),
    _metaData(lsst::daf::base::DataProperty::createPropertyNode("FitsMetaData")),
    _maskedImage(cols, rows),
    _wcsPtr(new lsst::afw::image::Wcs (wcs))
{}


/** @brief Construct a subExposure given an Exposure and a bounding box
  *
  * This addresses Ticket #111 (assigned to NMS on
  * 20070726).  The current implementation makes no effort to alter the existing
  * Wcs - a copy of the original Wcs is passed to the new subExposure.
  *
  * @throw a lsst::pex::exceptions::InvalidParameter if the requested subRegion
  * is not fully contained by the original MaskedImage BBox.
  */        
template<typename ImageT, typename MaskT, typename VarianceT> 
lsst::afw::image::Exposure<ImageT, MaskT, VarianceT>::Exposure(Exposure const &src,
                                                               Bbox const& bbox,
                                                               bool const deep) :
    lsst::daf::data::LsstBase(typeid(this)),
    _metaData(lsst::daf::base::DataProperty::createPropertyNode("FitsMetaData")),
    _maskedImage(src.getMaskedImage(), bbox, deep),
    _wcsPtr(new lsst::afw::image::Wcs(*this->_wcsPtr))
{}


/** @brief Construct an Image from FITS files.
  *
  * Take the Exposure's base input file name (as a std::string without
  * the _img.fits, _var.fits, or _msk.fits suffixes) and gets the MaskedImage of
  * the Exposure.  The method then uses the MaskedImage 'readFits' method to
  * read the MaskedImage of the Exposure and gets the Exposure's Wcs.
  *
  * @return the MaskedImage and the Wcs object with appropriate metadata of the
  * Exposure.
  *  
  * @note The method warns the user if the Exposure does not have a Wcs.
  *
  * @throw an lsst::pex::exceptions::NotFound if the MaskedImage could not be
  * read or the base file name could not be found.
  */
template<typename ImageT, typename MaskT, typename VarianceT> 
lsst::afw::image::Exposure<ImageT, MaskT, VarianceT>::Exposure(std::string const& expInFile, ///< Exposure's base input file name
                                                               int const hdu,                ///< Desired HDU
                                                               bool conformMasks  //!< Make Mask conform to mask layout in file?
                                                         ) :
    lsst::daf::data::LsstBase(typeid(this)),
    _metaData(lsst::daf::base::DataProperty::createPropertyNode("FitsMetaData")),
    _maskedImage(expInFile, hdu, _metaData, conformMasks),
    _wcsPtr(new lsst::afw::image::Wcs(_metaData))
{
    ;
}

/** Destructor
 */
template<typename ImageT, typename MaskT, typename VarianceT> 
lsst::afw::image::Exposure<ImageT, MaskT, VarianceT>::~Exposure(){}


/** @brief Get the Wcs of an Exposure.
  *
  * @return a copy of the boost::shared_ptr to the Wcs.
  *
  * @throw a lsst::pex::exceptions::NotFound if the Exposure does not have a Wcs.
  */
template<typename ImageT, typename MaskT, typename VarianceT> 
lsst::afw::image::Wcs lsst::afw::image::Exposure<ImageT, MaskT, VarianceT>::getWcs() const { 
    
    if (_wcsPtr.get() == 0) {
        throw lsst::pex::exceptions::NotFound("The Exposure does not have Wcs!!");
    }
    return *_wcsPtr;
}

// SET METHODS

/** @brief Set the MaskedImage of the Exposure.
  */   
template<typename ImageT, typename MaskT, typename VarianceT> 
void lsst::afw::image::Exposure<ImageT, MaskT, VarianceT>::setMaskedImage(lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> &maskedImage){
    _maskedImage = maskedImage; 
}


/** @brief Set the Wcs of the Exposure.  
 */   
template<typename ImageT, typename MaskT, typename VarianceT> 
void lsst::afw::image::Exposure<ImageT, MaskT, VarianceT>::setWcs(lsst::afw::image::Wcs const &wcs){
    _wcsPtr.reset(new lsst::afw::image::Wcs(wcs)); 
}


// Write FITS

/** @brief Write the Exposure's Image files.  Update the fits image header card
  * to reflect the Wcs information.
  *
  * Member takes the Exposure's base output file name (as a std::string without
  * the _img.fits, _var.fits, or _msk.fits suffixes) and uses the MaskedImage
  * Class to write the MaskedImage files, _img.fits, _var.fits, and _msk.fits to
  * disk.  Method also uses the metaData information to update the Exposure's
  * fits header cards.
  *
  * @note The MaskedImage Class will throw an pex Exception if the base
  * filename is not found.
  */
template<typename ImageT, typename MaskT, typename VarianceT> 
void lsst::afw::image::Exposure<ImageT, MaskT, VarianceT>::writeFits(
	const std::string &expOutFile ///< Exposure's base output file name
                                                                    ) const {
    lsst::daf::base::DataProperty::PtrType wcsDP = lsst::afw::formatters::WcsFormatter::generateDataProperty(*_wcsPtr);
    lsst::daf::base::DataProperty::PtrType
        outputMetaData(new lsst::daf::base::DataProperty(*_metaData)); // should be a deep copy

    _maskedImage.writeFits(expOutFile, outputMetaData);
}

// Explicit instantiations
template class lsst::afw::image::Exposure<boost::uint16_t>;
template class lsst::afw::image::Exposure<int>;
template class lsst::afw::image::Exposure<float>;
template class lsst::afw::image::Exposure<double>;
