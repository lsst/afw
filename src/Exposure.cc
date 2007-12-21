// -*- LSST-C++ -*- // fixed format comment for emacs
/**
  * \file Exposure.cc
  *
  * \ingroup fw
  *
  * \brief Implementation of the Exposure Class for LSST.  Class declaration in
  * Exposure.h.
  *
  * \author Nicole M. Silvestri, University of Washington
  *
  * Created on: Mon Apr 23 13:01:15 2007
  *
  * \version 
  *
  * LSST Legalese here... 
  */

#include <boost/cstdint.hpp> 
#include <boost/format.hpp> 
#include <boost/shared_ptr.hpp>

#include <vw/Image.h>
#include <vw/Math/BBox.h>

#include <stdexcept>

#include <lsst/mwi/data/LsstBase.h>
#include <lsst/mwi/exceptions.h>
#include <lsst/mwi/utils/Trace.h> 
#include <lsst/fw/Exposure.h>
#include <lsst/fw/MaskedImage.h>
#include <lsst/fw/WCS.h> 

/** \brief Exposure Class Implementation for LSST: a templated framework class
  * for creating an Exposure from a MaskedImage and a WCS.
  *
  * An Exposure is required to take one lsst::fw::MaskedImage or a region (col,
  * row) defining the size of a MaskedImage (this can be of size 0,0).  An
  * Exposure can (but is not required to) contain a lsst::fw::WCS.
  *
  * The template types should optimally be a float, double, unsigned int 16 bit,
  * or unsigned int 32 bit for the image (pixel) type and an unsigned int 32 bit
  * for the mask type.  These types have been explicitly instantiated for the
  * Exposure class.  All MaskedImage and WCS constructors are 'const' to allow
  * for views and copying.
  *
  * An Exposure can get and return its MaskedImage, WCS, and a subExposure.  The
  * getSubExposure member takes a vw::BBox region defining the subRegion of the
  * original Exposure to be returned.  The member retrieves the MaskedImage
  * corresponding to the subRegion.  The MaskedImage class throws an exception
  * for any subRegion extending beyond the original MaskedImage bounding
  * box. This member is not yet fully implemented because it requires the WCS
  * class to return the WCS metadata to the member so the CRPIX values of the
  * WCS can be adjusted to reflect the new subMaskedImage origin.  The
  * getSubExposure member will eventually return a subExposure consisting of the
  * subMAskedImage and the WCS object with its corresponding adjusted metadata.
  *
  * The hasWcs member is used to determine if the Exposure has a WCS.  It is not
  * required to have one.
  *
  * The read/writeFits members are not fully implemented yet.  They require both
  * persistence formatters and the metadata to be passed form the WCS class.
  *
  * TODO 20070905 Nicole M. Silvestri; By DC2: 
  *
  * 1. 'getSubExposure' method should deal with edge extension issues and at a
  * minimum, change the CRPIX (and other) values for the WCS in the metedata
  * header. Modify header keywords in the MaskedImage metadata (can use MI class
  * for this) and overwrite existing WCS kewords.  This should really be part of
  * the WCS class - discuss with TA. (for DC2)
  *
  * 2. Need to implement writeFits functionality...see persistance information
  * (boost::serialize - discuss w/ K-T & Jeff B.)  need formatter (see Greg D.'s
  * README on the Twiki.  Again need modifications to the WCS Class to get this
  * to work properly...same goes for the reafits method. (for DC2)
  *
  * 3. Test 'getSubExposure' method when complete and then close Ticket
  * #111. (for DC2)
  *
  * 4. Re-Sync with EA UML model and update robustness and seuqence
  * diagrams. (before CoDR)
  */

// CLASS CONSTRUCTORS and DESTRUCTOR

/** \brief Construct a blank Exposure of size 0x0 with no WCS. 
  */     
template<typename ImageT, typename MaskT> 
lsst::fw::Exposure<ImageT, MaskT>::Exposure(  
    ) : 
    lsst::mwi::data::LsstBase(typeid(this)),   
    _maskedImage(0,0),
    _wcsPtr()                 
{}


/** \brief Construct an Exposure without a WCS.
  */       
template<typename ImageT, typename MaskT> 
lsst::fw::Exposure<ImageT, MaskT>::Exposure( 
    lsst::fw::MaskedImage<ImageT, MaskT> const &maskedImage ///< the MaskedImage
    ) : 
    lsst::mwi::data::LsstBase(typeid(this)),   
    _maskedImage(maskedImage),
    _wcsPtr()
{}


/** \brief Construct an Exposure with a WCS.
  */               
template<typename ImageT, typename MaskT> 
lsst::fw::Exposure<ImageT, MaskT>::Exposure(
    lsst::fw::MaskedImage<ImageT, MaskT> const &maskedImage, ///< the MaskedImage
    lsst::fw::WCS const &wcs                                 ///< the WCS
    ) : 
    lsst::mwi::data::LsstBase(typeid(this)),
    _maskedImage(maskedImage),
    _wcsPtr(new lsst::fw::WCS (wcs))
{}


/** \brief Construct an Exposure with a blank MaskedImage of specified size and
  * a WCS.
  */          
template<typename ImageT, typename MaskT> 
lsst::fw::Exposure<ImageT, MaskT>::Exposure(
    unsigned cols,            ///< number of columns in the MaskedImage
    unsigned rows,            ///< number of rows in the MaskedImage
    lsst::fw::WCS const &wcs  ///< the WCS
    ) :
    lsst::mwi::data::LsstBase(typeid(this)),
    _maskedImage(cols, rows),
    _wcsPtr(new lsst::fw::WCS (wcs))
{}


/** \brief Construct an Exposure with a blank MaskedImage of specified size and
  *  without a WCS.
  */          
template<typename ImageT, typename MaskT> 
lsst::fw::Exposure<ImageT, MaskT>::Exposure(
    unsigned cols, ///< number of columns in the MaskedImage
    unsigned rows  ///< number of rows in the MaskedImage
    ) :
    lsst::mwi::data::LsstBase(typeid(this)),
    _maskedImage(cols, rows),
    _wcsPtr()
{}


/** Destructor
 */
template<typename ImageT, typename MaskT> 
lsst::fw::Exposure<ImageT, MaskT>::~Exposure(){}


/** \brief Get the WCS of an Exposure.
  *
  * \return a copy of the boost::shared_ptr to the WCS.
  *
  * \throw a lsst::mwi::exceptions::NotFound if the Exposure does not have a WCS.
  */
template<typename ImageT, typename MaskT> 
lsst::fw::WCS lsst::fw::Exposure<ImageT, MaskT>::getWcs() const { 
    
    if (_wcsPtr.get() == 0) {
        throw lsst::mwi::exceptions::NotFound("The Exposure does not have WCS!!");
    }
    return *_wcsPtr;
}


/** \brief Get a subExposure given an Exposure and a VW bounding box structure
  * (BBox2i) as the subRegion. This addresses Ticket #111 (assigned to NMS on
  * 20070726).  The current implementation makes no effort to alter the existing
  * WCS - a copy of the original WCS is passed to the new subExposure.
  *
  * \return the subExposure.
  * 
  * \throw a lsst::mwi::exceptions::InvalidParameter if the requested subRegion
  * is not fully contained by the original MaskedImage BBox.
  */        
template<typename ImageT, typename MaskT> 
lsst::fw::Exposure<ImageT, MaskT> lsst::fw::Exposure<ImageT, MaskT>::getSubExposure(const vw::BBox2i &subRegion ///< vw bounding box structure for sub-region 
) const {    

    typename lsst::fw::MaskedImage<ImageT, MaskT>::MaskedImagePtrT subMskImPtr = _maskedImage.getSubImage(subRegion);
    lsst::mwi::data::DataProperty::PtrType miMetaData = subMskImPtr->getImage()->getMetaData();
    lsst::fw::WCS miWcs(miMetaData);
    lsst::fw::Exposure<ImageT, MaskT> subExposure(*subMskImPtr, miWcs);
    return subExposure;
}


// SET METHODS

/** \brief Set the MaskedImage of the Exposure.
  */   
template<typename ImageT, typename MaskT> 
void lsst::fw::Exposure<ImageT, MaskT>::setMaskedImage(lsst::fw::MaskedImage<ImageT, MaskT> &maskedImage){
    _maskedImage = maskedImage; 
}


/** \brief Set the WCS of the Exposure.  
 */   
template<typename ImageT, typename MaskT> 
void lsst::fw::Exposure<ImageT, MaskT>::setWcs(lsst::fw::WCS const &wcs){
    _wcsPtr.reset(new lsst::fw::WCS(wcs)); 
}


// READ FITS AND WRITE FITS METHODS

/** \brief Read the Exposure's Image files.
  *
  * Member takes the Exposure's base input file name (as a std::string without
  * the _img.fits, _var.fits, or _msk.fits suffixes) and gets the MaskedImage of
  * the Exposure.  The method then uses the MaskedImage 'readFits' method to
  * read the MaskedImage of the Exposure and gets the Exposure's WCS.
  *
  * \return the MaskedImage and the WCS object with appropriate metadata of the
  * Exposure.
  *  
  * \note The method warns the user if the Exposure does not have a WCS.
  *
  * \throw an lsst::mwi::exceptions::NotFound if the MaskedImage could not be
  * read or the base file name could not be found.
  */
template<typename ImageT, typename MaskT> 
void lsst::fw::Exposure<ImageT, MaskT>::readFits(
    const std::string &expInFile ///< Exposure's base input file name
    ) {

    // really need the ability to construct an exposure from a string name this
    // only works if the input image is a MaskedImage.  MaskedImage class will
    // throw an exception otherwise.

     _maskedImage.readFits(expInFile);
     lsst::mwi::data::DataProperty::PtrType mData = _maskedImage.getImage()->getMetaData();
     lsst::fw::WCS newWcs(mData);
     _wcsPtr.reset(new lsst::fw::WCS(newWcs));
}


/** \brief Write the Exposure's Image files.  Update the fits image header card
  * to reflect the WCS information.
  *
  * Member takes the Exposure's base output file name (as a std::string without
  * the _img.fits, _var.fits, or _msk.fits suffixes) and uses the MaskedImage
  * Class to write the MaskedImage files, _img.fits, _var.fits, and _msk.fits to
  * disk.  Method also uses the metaData information to update the Exposure's
  * fits header cards.
  *
  * \note The MaskedImage Class will throw an mwi Exception if the base
  * filename is not found.
  */
template<typename ImageT, typename MaskT> 
void lsst::fw::Exposure<ImageT, MaskT>::writeFits(
    const std::string &expOutFile ///< Exposure's base output file name
    ) const {

//    std::string suffix = ".fits";
    //   std::string expFile = expOutFile + suffix;

    // Since this is not yet implemented to work properly - throw an exception

//     lsst::fw::Image<ImageT> _image;
//     _image->writefits(expFile);

    throw lsst::mwi::exceptions::InvalidParameter("'writeFits' method has not been implemented!");
    
    // eventually need to add functionality here...
    
}

// Explicit instantiations
template class lsst::fw::Exposure<float, lsst::fw::maskPixelType>;
template class lsst::fw::Exposure<double, lsst::fw::maskPixelType>;
template class lsst::fw::Exposure<boost::uint16_t, lsst::fw::maskPixelType>;
