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

#include "lsst/daf/base/PropertySet.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Exposure.h"
#include "lsst/afw/formatters/WcsFormatter.h"

namespace afwImage = lsst::afw::image;

/** @brief Exposure Class Implementation for LSST: a templated framework class
  * for creating an Exposure from a MaskedImage and a Wcs.
  *
  * An Exposure is required to take one afwImage::MaskedImage or a region (col,
  * row) defining the size of a MaskedImage (this can be of size 0,0).  An
  * Exposure can (but is not required to) contain a afwImage::Wcs.
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


/** @brief Construct an Exposure with a blank MaskedImage of specified size (default 0x0) and
  * a Wcs (which may be default constructed)
  */          
template<typename ImageT, typename MaskT, typename VarianceT> 
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(int cols, ///< number of columns (default: 0)
                                                               int rows, ///< number of rows (default: 0)
                                                               afwImage::Wcs const& wcs ///< the Wcs
                                                              ) :
    lsst::daf::data::LsstBase(typeid(this)),
    _maskedImage(cols, rows),
    _wcs(new afwImage::Wcs(wcs))
{
    setMetadata(lsst::daf::base::PropertySet::Ptr(new lsst::daf::base::PropertySet()));
}

/** @brief Construct an Exposure from a MaskedImage
  */               
template<typename ImageT, typename MaskT, typename VarianceT> 
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    MaskedImageT &maskedImage, ///< the MaskedImage
    afwImage::Wcs const& wcs                                      ///< the Wcs
                                                              ) :
    lsst::daf::data::LsstBase(typeid(this)),
    _maskedImage(maskedImage),
    _wcs(new afwImage::Wcs(wcs))
{
    setMetadata(lsst::daf::base::PropertySet::Ptr(new lsst::daf::base::PropertySet()));
}

/** @brief Construct a subExposure given an Exposure and a bounding box
  *
  * @throw a lsst::pex::exceptions::InvalidParameter if the requested subRegion
  * is not fully contained by the original MaskedImage BBox.
  */        
template<typename ImageT, typename MaskT, typename VarianceT> 
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(Exposure const &src, ///< Parent Exposure
                                                       BBox const& bbox,    ///< Desired region in Exposure 
                                                       bool const deep      ///< Should we copy the pixels?
                                                              ) :
    lsst::daf::data::LsstBase(typeid(this)),
    _maskedImage(src.getMaskedImage(), bbox, deep),
    _wcs(new afwImage::Wcs(*src._wcs))
{
    _wcs->shiftReferencePixel(-bbox.getX0(), -bbox.getY0());
    setMetadata(lsst::daf::base::PropertySet::Ptr(new lsst::daf::base::PropertySet()));
}

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
 * @note We use FITS numbering, so the first HDU is HDU 1, not 0 (although we're helpful and interpret 0 as meaning
 * the first HDU, i.e. HDU 1).  I.e. if you have a PDU, the numbering is thus [PDU, HDU2, HDU3, ...]
 *
 * @throw an lsst::pex::exceptions::NotFound if the MaskedImage could not be
 * read or the base file name could not be found.
 */
template<typename ImageT, typename MaskT, typename VarianceT> 
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    std::string const& baseName,    ///< Exposure's base input file name
    int const hdu,                  ///< Desired HDU
    BBox const& bbox,               //!< Only read these pixels
    bool conformMasks               //!< Make Mask conform to mask layout in file?
) :
    lsst::daf::data::LsstBase(typeid(this)) {
    lsst::daf::base::PropertySet::Ptr metadata(new lsst::daf::base::PropertySet());

    _maskedImage = MaskedImageT(baseName, hdu, metadata, bbox, conformMasks);

    if (bbox) {
        try {
            metadata->set("CRPIX1", metadata->getAsDouble("CRPIX1") - bbox.getX0());
            metadata->set("CRPIX2", metadata->getAsDouble("CRPIX2") - bbox.getY0());
        }
        catch (lsst::pex::exceptions::NotFoundException &e) {
            ; // OK, no WCS is present in header
        }
    }

    _wcs = afwImage::Wcs::Ptr(new afwImage::Wcs(metadata));
    setMetadata(metadata);
}

/** Destructor
 */
template<typename ImageT, typename MaskT, typename VarianceT> 
afwImage::Exposure<ImageT, MaskT, VarianceT>::~Exposure(){}


/** @brief Get the Wcs of an Exposure.
  *
  * @return a boost::shared_ptr to the Wcs.
  */
template<typename ImageT, typename MaskT, typename VarianceT> 
afwImage::Wcs::Ptr afwImage::Exposure<ImageT, MaskT, VarianceT>::getWcs() const { 
    return _wcs;
}

// SET METHODS

/** @brief Set the MaskedImage of the Exposure.
  */   
template<typename ImageT, typename MaskT, typename VarianceT> 
void afwImage::Exposure<ImageT, MaskT, VarianceT>::setMaskedImage(MaskedImageT &maskedImage){
    _maskedImage = maskedImage; 
}


/** @brief Set the Wcs of the Exposure.  
 */   
template<typename ImageT, typename MaskT, typename VarianceT> 
void afwImage::Exposure<ImageT, MaskT, VarianceT>::setWcs(afwImage::Wcs const &wcs){
    _wcs.reset(new afwImage::Wcs(wcs)); 
}


// Write FITS

/** @brief Write the Exposure's Image files.  Update the fits image header card
  * to reflect the Wcs information.
  *
  * Member takes the Exposure's base output file name (as a std::string without
  * the _img.fits, _var.fits, or _msk.fits suffixes) and uses the MaskedImage
  * Class to write the MaskedImage files, _img.fits, _var.fits, and _msk.fits to
  * disk.  Method also uses the metadata information to update the Exposure's
  * fits header cards.
  *
  * @note The MaskedImage Class will throw an pex Exception if the base
  * filename is not found.
  */
template<typename ImageT, typename MaskT, typename VarianceT> 
void afwImage::Exposure<ImageT, MaskT, VarianceT>::writeFits(
    const std::string &expOutFile ///< Exposure's base output file name
) const {
    using lsst::daf::base::PropertySet;

    PropertySet::Ptr outputMetadata = getMetadata()->deepCopy();
    PropertySet::Ptr wcsMetadata = lsst::afw::formatters::WcsFormatter::generatePropertySet(*_wcs);
    //
    // Copy wcsMetadata over to outputMetadata
    //
    outputMetadata->combine(wcsMetadata);

    _maskedImage.writeFits(expOutFile, outputMetadata);
}

// Explicit instantiations
template class afwImage::Exposure<boost::uint16_t>;
template class afwImage::Exposure<int>;
template class afwImage::Exposure<float>;
template class afwImage::Exposure<double>;
