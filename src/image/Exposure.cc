// -*- LSST-C++ -*- // fixed format comment for emacs

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
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
  */

#include <stdexcept>

#include "boost/cstdint.hpp" 
#include "boost/format.hpp" 
#include "boost/shared_ptr.hpp"
#include "boost/algorithm/string/trim.hpp"

#include "lsst/daf/base/PropertySet.h"
#include "lsst/daf/base/PropertyList.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/image/Exposure.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/image/Calib.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/cameraGeom/Detector.h"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace afwDetection = lsst::afw::detection;
namespace cameraGeom = lsst::afw::cameraGeom;

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
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    unsigned int width,                 ///< number of columns
    unsigned int height,                ///< number of rows
    CONST_PTR(Wcs) wcs        ///< the Wcs
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(width, height),
    _info(new ExposureInfo(wcs))
{}

/** @brief Construct an Exposure with a blank MaskedImage of specified size (default 0x0) and
  * a Wcs (which may be default constructed)
  */          
template<typename ImageT, typename MaskT, typename VarianceT> 
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    afwGeom::Extent2I const & dimensions, ///< desired image width/height
    CONST_PTR(Wcs) wcs          ///< the Wcs
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(dimensions),
    _info(new ExposureInfo(wcs))
{}

/** @brief Construct an Exposure with a blank MaskedImage of specified size (default 0x0) and
  * a Wcs (which may be default constructed)
  */          
template<typename ImageT, typename MaskT, typename VarianceT> 
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    afwGeom::Box2I const & bbox, ///< desired image width/height, and origin
    CONST_PTR(Wcs) wcs ///< the Wcs
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(bbox),
    _info(new ExposureInfo(wcs))
{}

/** @brief Construct an Exposure from a MaskedImage
  */               
template<typename ImageT, typename MaskT, typename VarianceT> 
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    MaskedImageT &maskedImage, ///< the MaskedImage
    CONST_PTR(Wcs) wcs  ///< the Wcs
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(maskedImage),
    _info(new ExposureInfo(wcs))
{}


/** @brief Copy an Exposure
  */        
template<typename ImageT, typename MaskT, typename VarianceT> 
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    Exposure const &src, ///< Parent Exposure
    bool const deep      ///< Should we copy the pixels?
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(src.getMaskedImage(), deep),
    _info(new ExposureInfo(*src.getInfo()))
{}

/** @brief Construct a subExposure given an Exposure and a bounding box
  *
  * @throw a lsst::pex::exceptions::InvalidParameter if the requested subRegion
  * is not fully contained by the original MaskedImage BBox.
  */        
template<typename ImageT, typename MaskT, typename VarianceT> 
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    Exposure const &src, ///< Parent Exposure
    afwGeom::Box2I const& bbox,    ///< Desired region in Exposure 
    ImageOrigin const origin,   ///< Coordinate system for bbox
    bool const deep      ///< Should we copy the pixels?
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(src.getMaskedImage(), bbox, origin, deep),
    _info(new ExposureInfo(*src.getInfo()))
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
    afwGeom::Box2I const& bbox,               //!< Only read these pixels
    ImageOrigin const origin,       ///< Coordinate system for bbox
    bool conformMasks               //!< Make Mask conform to mask layout in file?
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(),
    _info(new ExposureInfo())
{
    lsst::daf::base::PropertySet::Ptr metadata(new lsst::daf::base::PropertyList());

    _maskedImage = MaskedImageT(baseName, hdu, metadata, bbox, origin, conformMasks);
    
    postFitsCtorInit(metadata);
}

/** Destructor
 */
template<typename ImageT, typename MaskT, typename VarianceT> 
afwImage::Exposure<ImageT, MaskT, VarianceT>::~Exposure(){}

/**
Finish initialization after constructing from a FITS file
*/
template<typename ImageT, typename MaskT, typename VarianceT> 
void afwImage::Exposure<ImageT, MaskT, VarianceT>::postFitsCtorInit(
    lsst::daf::base::PropertySet::Ptr metadata
) {
    // true: strip keywords that are related to the created WCS from the input
    // metadata
    _info->setWcs(afwImage::makeWcs(metadata, true));
    /*
     * Filter
     */
    _info->setFilter(Filter(metadata, true));
    afwImage::detail::stripFilterKeywords(metadata);
    /*
     * Calib
     */
    PTR(afwImage::Calib) newCalib(new afwImage::Calib(metadata));
    _info->setCalib(newCalib);
    afwImage::detail::stripCalibKeywords(metadata);
    /*
     * Set the remaining parts of the metadata
     */
    _info->setMetadata(metadata);
}

// SET METHODS

/** @brief Set the MaskedImage of the Exposure.
  */   
template<typename ImageT, typename MaskT, typename VarianceT> 
void afwImage::Exposure<ImageT, MaskT, VarianceT>::setMaskedImage(MaskedImageT &maskedImage){
    _maskedImage = maskedImage; 
}

template<typename ImageT, typename MaskT, typename VarianceT> 
void afwImage::Exposure<ImageT, MaskT, VarianceT>::setXY0(afwGeom::Point2I const& origin) {
    afwGeom::Point2I old(_maskedImage.getXY0());
    if (_info->hasWcs())
        _info->getWcs()->shiftReferencePixel(origin.getX() - old.getX(), origin.getY() - old.getY());
    _maskedImage.setXY0(origin);
}


// Write FITS

/** @brief Save the Exposure and all its components to a multi-extension FITS file.
 *
 * @note LSST and FITS use a different convention for WCS coordinates.
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
 */
template<typename ImageT, typename MaskT, typename VarianceT> 
void afwImage::Exposure<ImageT, MaskT, VarianceT>::writeFits(
    std::string const & fileName ///< Exposure's output file name
) const {
    fits::Fits fitsfile(fileName, "w", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile);
}

/**
 * See writeFits(string) for a basic description of this function.
 *
 * This function differs from the string version in that rather than writing a FITS file to disk
 * it writes a FITS file to a RAM buffer.
 */
template<typename ImageT, typename MaskT, typename VarianceT> 
void afwImage::Exposure<ImageT, MaskT, VarianceT>::writeFits(
    fits::MemFileManager & manager
) const {
    fits::Fits fitsfile(manager, "w", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile);
}

template<typename ImageT, typename MaskT, typename VarianceT> 
void afwImage::Exposure<ImageT, MaskT, VarianceT>::writeFits(
    fits::Fits & fitsfile
) const {
    lsst::daf::base::PropertySet::Ptr outputMetadata = _info->getFitsMetadata(getXY0());
    _maskedImage.writeFits(fitsfile, outputMetadata);
}

// Explicit instantiations
/// \cond
template class afwImage::Exposure<boost::uint16_t>;
template class afwImage::Exposure<int>;
template class afwImage::Exposure<float>;
template class afwImage::Exposure<double>;
template class afwImage::Exposure<boost::uint64_t>;
/// \endcond
