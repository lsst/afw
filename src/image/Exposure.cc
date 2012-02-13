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
  *
  * @version 
  *
  * LSST Legalese here... 
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
#include "lsst/afw/image/Exposure.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/image/Calib.h"
#include "lsst/afw/fits.h"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace afwDetection = lsst::afw::detection;
namespace afwFits = lsst::afw::fits;

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
    afwImage::Wcs const & wcs           ///< the Wcs
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(width, height),
    _wcs(wcs.clone()),
    _detector(),
    _filter(),
    _calib(new afwImage::Calib())
{
    setMetadata(lsst::daf::base::PropertySet::Ptr(new lsst::daf::base::PropertyList()));
}

/** @brief Construct an Exposure with a blank MaskedImage of specified size (default 0x0) and
  * a Wcs (which may be default constructed)
  */          
template<typename ImageT, typename MaskT, typename VarianceT> 
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    afwGeom::Extent2I const & dimensions, ///< desired image width/height
    afwImage::Wcs const & wcs   ///< the Wcs
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(dimensions),
    _wcs(wcs.clone()),
    _detector(),
    _filter(),
    _calib(new afwImage::Calib())
{
    setMetadata(lsst::daf::base::PropertySet::Ptr(new lsst::daf::base::PropertyList()));
}

/** @brief Construct an Exposure with a blank MaskedImage of specified size (default 0x0) and
  * a Wcs (which may be default constructed)
  */          
template<typename ImageT, typename MaskT, typename VarianceT> 
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    afwGeom::Box2I const & bbox, ///< desired image width/height, and origin
    afwImage::Wcs const & wcs   ///< the Wcs
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(bbox),
    _wcs(wcs.clone()),
    _detector(),
    _filter(),
    _calib(new afwImage::Calib())
{
    setMetadata(lsst::daf::base::PropertySet::Ptr(new lsst::daf::base::PropertyList()));
}

/** @brief Construct an Exposure from a MaskedImage
  */               
template<typename ImageT, typename MaskT, typename VarianceT> 
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    MaskedImageT &maskedImage, ///< the MaskedImage
    afwImage::Wcs const& wcs   ///< the Wcs
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(maskedImage),
    _wcs(wcs.clone()),
    _detector(),
    _filter(),
    _calib(new afwImage::Calib()),
    _psf(PTR(lsst::afw::detection::Psf)())
{
    setMetadata(lsst::daf::base::PropertySet::Ptr(new lsst::daf::base::PropertyList()));
}


/** @brief Copy an Exposure
  */        
template<typename ImageT, typename MaskT, typename VarianceT> 
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    Exposure const &src, ///< Parent Exposure
    bool const deep      ///< Should we copy the pixels?
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(src.getMaskedImage(), deep),
    _wcs(src._wcs->clone()),
    _detector(src._detector),
    _filter(src._filter),
    _calib(new lsst::afw::image::Calib(*src.getCalib())),
    _psf(_clonePsf(src.getPsf()))
{
/*
  * N.b. You'll need to update the subExposure cctor and the generalised cctor in Exposure.h
  * when you add new data members --- this note is here as you'll be making the same changes here!
  */
    setMetadata(deep ? src.getMetadata()->deepCopy() : src.getMetadata());
}

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
    _wcs(src._wcs->clone()),
    _detector(src._detector),
    _filter(src._filter),
    _calib(new lsst::afw::image::Calib(*src.getCalib())),
    _psf(_clonePsf(src.getPsf()))
{
    setMetadata(deep ? src.getMetadata()->deepCopy() : src.getMetadata());
}

template<typename ImageT, typename MaskT, typename VarianceT> 
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    std::string const& baseName,
    int const hdu,
    afwGeom::Box2I const& bbox,
    ImageOrigin const origin,
    bool conformMasks
) :
    lsst::daf::base::Citizen(typeid(this))
{
    lsst::daf::base::PropertySet::Ptr metadata(new lsst::daf::base::PropertyList());
    _maskedImage = MaskedImageT(baseName, hdu, metadata, bbox, origin, conformMasks);    
    postFitsCtorInit(metadata);
}

template<typename ImageT, typename MaskT, typename VarianceT> 
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    lsst::afw::fits::MemFileManager & manager,
    int const hdu,
    afwGeom::Box2I const& bbox,
    ImageOrigin const origin,
    bool conformMasks
) :
    lsst::daf::base::Citizen(typeid(this))
{
    lsst::daf::base::PropertySet::Ptr metadata(new lsst::daf::base::PropertySet());
    _maskedImage = MaskedImageT(manager, hdu, metadata, bbox, origin, conformMasks);
    postFitsCtorInit(metadata);
}

template<typename ImageT, typename MaskT, typename VarianceT> 
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    lsst::afw::fits::Fits & fitsfile,
    afwGeom::Box2I const& bbox,
    ImageOrigin const origin,
    bool conformMasks
) :
    lsst::daf::base::Citizen(typeid(this))
{
    lsst::daf::base::PropertySet::Ptr metadata(new lsst::daf::base::PropertySet());
    _maskedImage = MaskedImageT(fitsfile, metadata, bbox, origin, conformMasks);
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
    _wcs = afwImage::Wcs::Ptr(afwImage::makeWcs(metadata));
    //
    // Strip keywords from the input metadata that are related to the generated Wcs
    //
    detail::stripWcsKeywords(metadata, _wcs);
    /*
     * Filter
     */
    _filter = Filter(metadata, true);
    afwImage::detail::stripFilterKeywords(metadata);
    /*
     * Calib
     */
    _calib = PTR(afwImage::Calib)(new afwImage::Calib(metadata));
    afwImage::detail::stripCalibKeywords(metadata);

    //If keywords LTV[1,2] are present, the image on disk is already a subimage, so
    //we should note this fact. Also, shift the wcs so the crpix values refer to 
    //pixel positions not pixel index
    //See writeFits() below
    std::string key = "LTV1";
    if( metadata->exists(key)) {
        _wcs->shiftReferencePixel(-1*metadata->getAsDouble(key), 0);
        metadata->remove(key);
    }
    key = "LTV2";
    if( metadata->exists(key) ) {
        _wcs->shiftReferencePixel(0, -1*metadata->getAsDouble(key));
        metadata->remove(key);
    }
    /*
     * Set the remaining parts of the metadata
     */
    setMetadata(metadata);
}

/**
 * Clone a Psf; defined here so that we don't have to expose the insides of Psf in Exposure.h
 */
template<typename ImageT, typename MaskT, typename VarianceT> 
PTR(afwDetection::Psf) afwImage::Exposure<ImageT, MaskT, VarianceT>::_clonePsf(
    CONST_PTR(afwDetection::Psf) psf      // the Psf to clone
) {
    return (psf) ? psf->clone() : PTR(afwDetection::Psf)();
}

/**
 * Clone a Calib; defined here so that we don't have to expose the insides of Calib in Exposure.h
 */
template<typename ImageT, typename MaskT, typename VarianceT> 
PTR(afwImage::Calib) afwImage::Exposure<ImageT, MaskT, VarianceT>::_cloneCalib(
    CONST_PTR(afwImage::Calib) calib    // the Calib to clone
) {
    return PTR(afwImage::Calib)(calib ? new afwImage::Calib(*calib) : NULL);
}

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
    _wcs = wcs.clone();
}


// Write FITS

template<typename ImageT, typename MaskT, typename VarianceT> 
lsst::daf::base::PropertySet::Ptr afwImage::Exposure<ImageT, MaskT, VarianceT>::generateOutputMetadata() const {
    using lsst::daf::base::PropertySet;
    
    //LSST convention is that Wcs is in pixel coordinates (i.e relative to bottom left
    //corner of parent image, if any). The Wcs/Fits convention is that the Wcs is in
    //image coordinates. When saving an image we convert from pixel to index coordinates.
    //In the case where this image is a parent image, the reference pixels are unchanged
    //by this transformation
    afwImage::MaskedImage<ImageT> mi = getMaskedImage();

    afwImage::Wcs::Ptr newWcs = _wcs->clone(); //Create a copy
    newWcs->shiftReferencePixel(-1*mi.getX0(), -1*mi.getY0() );

    //Create fits header
    PropertySet::Ptr outputMetadata = getMetadata()->deepCopy();
    // Copy wcsMetadata over to fits header
    PropertySet::Ptr wcsMetadata = newWcs->getFitsMetadata();
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
    if (_detector) {
        outputMetadata->set("DETNAME", _detector->getId().getName());
        outputMetadata->set("DETSER", _detector->getId().getSerial());
    }
    /**
     * We need to define these keywords properly! XXX
     */
    outputMetadata->set("TIME-MID", _calib->getMidTime().toString());
    outputMetadata->set("EXPTIME", _calib->getExptime());
    outputMetadata->set("FLUXMAG0", _calib->getFluxMag0().first);
    outputMetadata->set("FLUXMAG0ERR", _calib->getFluxMag0().second);
    
    return outputMetadata;
}

template<typename ImageT, typename MaskT, typename VarianceT> 
void afwImage::Exposure<ImageT, MaskT, VarianceT>::writeFits(afwFits::Fits & fitsfile) const {
    lsst::daf::base::PropertySet::Ptr outputMetadata = generateOutputMetadata();
    _maskedImage.writeFits(fitsfile, outputMetadata);
}

template<typename ImageT, typename MaskT, typename VarianceT> 
void afwImage::Exposure<ImageT, MaskT, VarianceT>::writeFits(
    std::string const & fileName, std::string const & mode
) const {
    lsst::daf::base::PropertySet::Ptr outputMetadata = generateOutputMetadata();
    _maskedImage.writeFits(fileName, outputMetadata, mode);
}

template<typename ImageT, typename MaskT, typename VarianceT> 
void afwImage::Exposure<ImageT, MaskT, VarianceT>::writeFits(
    afwFits::MemFileManager & manager, std::string const & mode
) const {
    lsst::daf::base::PropertySet::Ptr outputMetadata = generateOutputMetadata();
    _maskedImage.writeFits(manager, outputMetadata, mode);
}

// Explicit instantiations
/// \cond
template class afwImage::Exposure<boost::uint16_t>;
template class afwImage::Exposure<int>;
template class afwImage::Exposure<float>;
template class afwImage::Exposure<double>;
template class afwImage::Exposure<boost::uint64_t>;
/// \endcond
