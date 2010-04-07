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
#include "lsst/pex/logging/Trace.h"
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
    _wcs(new afwImage::Wcs(wcs)),
    _detector(),
    _filter()
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
    _wcs(new afwImage::Wcs(wcs)),
    _detector(),
    _filter()
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
    _wcs(new afwImage::Wcs(*src._wcs)),
    _detector(),
    _filter()    
{
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

    
    //Offset of origin of subexposure from origin of parent image (if any)
    PointI offset(0,0);
//     if(bbox) {
//         offset = offset + bbox.getLLC();
//     }

    _maskedImage = MaskedImageT(baseName, hdu, metadata, bbox, conformMasks);
    _wcs = afwImage::Wcs::Ptr(afwImage::makeWcs(metadata));

    //If keywords LTV[1,2] are present, the image on disk is already a subimage, so
    //we should note this fact. Also, shift the wcs so the crpix values refer to 
    //pixel positions not pixel index
    //See writeFits() below 
    if( metadata->exists("LTV1") ) {
        _wcs->shiftReferencePixel(-1*metadata->getAsDouble("LTV1"), 0);
    }
    if( metadata->exists("LTV2") ) {
        _wcs->shiftReferencePixel(0, -1*metadata->getAsDouble("LTV2"));
    }

    if( metadata->exists("FILTER") ) {
        std::string filterName = metadata->getAsString("FILTER");
        try {
            _filter = Filter(filterName);
        } catch(lsst::pex::exceptions::NotFoundException &) {
            lsst::pex::logging::TTrace<3>("afw.image.exposure", "Unknown filter %s", filterName.c_str());
        }
    }

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
  * @note LSST and Fits use a different convention for Wcs coordinates.
  * Fits measures crpix relative to the bottom left hand corner of the image
  * saved in that file (what ds9 calls image coordinates). Lsst measures it 
  * relative to the bottom left hand corner of the parent image (what 
  * ds9 calls the physical coordinates). This may cause confusion when you
  * write an image to disk and discover that the values of crpix in the header
  * are not what you expect.
  *
  * exposure = afwImage.ExposureF(filename) 
  * fitsHeader = afwImage.readMetadata(filename)
  * 
  * exposure.getWcs().getPixelOrigin() ---> (128,128)
  * fitsHeader.get("CRPIX1") --> 108
  *
  * This is expected. If you look at the value of
  * fitsHeader.get("LTV1") --> -20
  * you will find that CRPIX - LTV == getPixelOrigin.
  *
  * This implementation means that if you open the image in ds9 (say)
  * the wcs translations for a given pixel are correct
  *
  * @note The MaskedImage Class will throw an pex Exception if the base
  * filename is not found.
  */
template<typename ImageT, typename MaskT, typename VarianceT> 
void afwImage::Exposure<ImageT, MaskT, VarianceT>::writeFits(
    const std::string &expOutFile ///< Exposure's base output file name
) const {
    using lsst::daf::base::PropertySet;


    //LSST convention is that Wcs is in pixel coordinates (i.e relative to bottom left
    //corner of parent image, if any). The Wcs/Fits convention is that the Wcs is in
    //image coordinates. When saving an image we convert from pixel to index coordinates.
    //In the case where this image is a parent image, the reference pixels are unchanged
    //by this transformation
    afwImage::MaskedImage<ImageT> mi = getMaskedImage();

    afwImage::Wcs::Ptr newWcs(new afwImage::Wcs(*_wcs)); //Create a copy
    newWcs->shiftReferencePixel(-1*mi.getX0(), -1*mi.getY0() );

    //Create fits header
    PropertySet::Ptr outputMetadata = getMetadata()->deepCopy();
    // Copy wcsMetadata over to fits header
    PropertySet::Ptr wcsMetadata = lsst::afw::formatters::WcsFormatter::generatePropertySet(*newWcs);
    outputMetadata->combine(wcsMetadata);
    
    //Store _x0 and _y0. If this exposure is a portion of a larger image, _x0 and _y0
    //indicate the origin (the position of the bottom left corner) of the sub-image with 
    //respect to the origin of the parent image.
    //This is stored in the fits header using the LTV convention used by STScI 
    //(see \S2.6.2 of HST Data Handbook for STIS, version 5.0
    // http://www.stsci.edu/hst/stis/documents/handbooks/currentDHB/ch2_stis_data7.html#429287). 
    //This is not a fits standard keyword, but is recognised by ds9
    //LTV keywords use the opposite convention to the LSST, in that they represent
    //the position of the origin of the parent image relative to the origin of the sub-image.
    // _x0, _y0 >= 0, while LTV1 and LTV2 <= 0
  
    outputMetadata->set("LTV1", -1*mi.getX0());
    outputMetadata->set("LTV2", -1*mi.getY0());

    outputMetadata->set("FILTER", _filter.getName());
        
    _maskedImage.writeFits(expOutFile, outputMetadata);
}

// Explicit instantiations
template class afwImage::Exposure<boost::uint16_t>;
template class afwImage::Exposure<int>;
template class afwImage::Exposure<float>;
template class afwImage::Exposure<double>;
