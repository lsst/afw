// -*- lsst-c++ -*-
// Implementations of MaskedImage class methods
// This file can NOT be separately compiled!   It is included by MaskedImage.h

#include <typeinfo>
#include <sys/stat.h>
#include <lsst/pex/utils/Trace.h>
#include <lsst/pex/exceptions.h>

/**
 * \brief Construct an empty MaskedImage of size 0x0
 */
template<typename ImagePixelT, typename MaskPixelT> 
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::MaskedImage(MaskPlaneDict planeDefs) :
    lsst::daf::data::LsstBase(typeid(this)),
    _imagePtr(new Image<ImagePixelT>()),
    _variancePtr(new Image<ImagePixelT>()),
    _maskPtr(new Mask<MaskPixelT>(planeDefs)) {
}

/**
 * \brief Construct from a supplied Image and Mask. The Variance will be set to zero.
 */
template<typename ImagePixelT, typename MaskPixelT> 
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::MaskedImage(ImagePtrT image, MaskPtrT mask) :
    lsst::daf::data::LsstBase(typeid(this)),
    _imagePtr(image),
    _variancePtr(new Image<ImagePixelT>()),
    _maskPtr(mask) {
    conformSizes();
}    

/**
 * \brief Construct from a supplied Image, Variance, and Mask.
 */
template<typename ImagePixelT, typename MaskPixelT> 
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::MaskedImage(
    ImagePtrT image, ImagePtrT variance, MaskPtrT mask
) :
    lsst::daf::data::LsstBase(typeid(this)),
    _imagePtr(image),
    _variancePtr(variance),
    _maskPtr(mask) {
    conformSizes();
}

/**
 * \brief Construct a blank MaskedImage of specified size
 */
template<typename ImagePixelT, typename MaskPixelT> 
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::MaskedImage(int nCols, int nRows, MaskPlaneDict planeDefs) :
    lsst::daf::data::LsstBase(typeid(this)),
    _imagePtr(new Image<ImagePixelT>(nCols, nRows)),
    _variancePtr(new Image<ImagePixelT>(nCols, nRows)),
    _maskPtr(new Mask<MaskPixelT>(nCols, nRows, planeDefs)) {
}

/**
 * \brief Copy constructor
 *
 * Warning: this is a shallow copy; the pixel data is shared with the original MaskedImage.
 *
 * I would not expect an explicit assignment operator to be necessary, given the use of shared pointers,
 * but it allows lsst::afw::convolveLinear to function properly.
 */
template<typename ImagePixelT, typename MaskPixelT>
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::MaskedImage(
    const lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>& rhs
) :
    lsst::daf::data::LsstBase(typeid(this)),
    _imagePtr(rhs._imagePtr),
    _variancePtr(rhs._variancePtr),
    _maskPtr(rhs._maskPtr) {
}

template<typename ImagePixelT, typename MaskPixelT> 
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::~MaskedImage() {
}

/**
 * \brief Assignment operator
 *
 * Warning: this is a shallow copy; the pixel data is shared with the original MaskedImage.
 *
 * I would not expect an explicit assignment operator to be necessary, given the use of shared pointers,
 * but it avoids memory problems (see ticket 144).
 */
template<typename ImagePixelT, typename MaskPixelT>
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>& lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::operator= (
    const MaskedImage<ImagePixelT, MaskPixelT>& rhs
) {
    if (&rhs != this) {   // beware of self assignment: maskedImage = maskedImage;
        _imagePtr.reset();
        _variancePtr.reset();
        _maskPtr.reset();
        _imagePtr = rhs._imagePtr;
        _variancePtr = rhs._variancePtr;
        _maskPtr = rhs._maskPtr;
    }
    
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT>
inline typename lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::ImagePtrT
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::getImage() const {
    return _imagePtr;
}

template<typename ImagePixelT, typename MaskPixelT>
inline typename lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::ImagePtrT
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::getVariance() const {
    return _variancePtr;
}

template<typename ImagePixelT, typename MaskPixelT>
inline typename lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::MaskPtrT
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::getMask() const {
    return _maskPtr;
}

/**
 * \brief Read a masked image from a trio of FITS files
 *
 * Image data is loaded from (baseName)_img.fits
 * Variance is loaded from (baseName)_var.fits, if found
 * Mask data is loaded from (baseName)_msk.fits, if found
 *
 * \throw lsst::pex::exceptions::NotFound if none of (baseName){_img,_var,_msk}.fits is found
 */
template<typename ImagePixelT, typename MaskPixelT>
void lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::readFits(std::string baseName, bool conformMaskPlanes) {

    const std::string imageSuffix = "_img.fits";
    const std::string maskSuffix = "_msk.fits";
    const std::string varianceSuffix = "_var.fits";

    bool fileFound = false;

// reset any existing data

    typename Image<ImagePixelT>::ImageIVwPtrT _imageVwPtr = _imagePtr->getIVwPtr();
    typename Image<ImagePixelT>::ImageIVwPtrT _varianceVwPtr = _variancePtr->getIVwPtr();
    typename Mask<MaskPixelT>::MaskIVwPtrT _maskVwPtr = _maskPtr->getIVwPtr();

    _imageRows = 0;
    _imageCols = 0;
    _imageVwPtr->set_size(0,0);
    _varianceVwPtr->set_size(0,0);
    _maskVwPtr->set_size(0,0);

    struct stat statFileInfo;
    std::string imageFileName = baseName + imageSuffix;
    if (stat(imageFileName.c_str(), &statFileInfo) == 0) {
       fileFound = true;
       _imagePtr->readFits(imageFileName);
    }

    std::string varianceFileName = baseName + varianceSuffix;
    if (stat(varianceFileName.c_str(), &statFileInfo) == 0) {
       fileFound = true;
       _variancePtr->readFits(varianceFileName);
    }

    std::string maskFileName = baseName + maskSuffix;
    if (stat(maskFileName.c_str(), &statFileInfo) == 0) {
        fileFound = true;
        _maskPtr->readFits(maskFileName, conformMaskPlanes);
    }

//  if no file found, throw an exception

    if (fileFound == false) {
        throw lsst::pex::exceptions::NotFound(boost::format("Failed to open %s{%s, %s, %s}") %
                       baseName % imageSuffix % varianceSuffix % maskSuffix);
    }

//  ensure all image components have the same size.  set_size is a nop if size would be unchanged

    conformSizes();

    lsst::pex::utils::Trace("afw.MaskedImage", 2,
              boost::format("Read in MaskedImage of size (%d,%d)") % _imageCols % _imageRows);

}

/**
 * \brief Write a masked image to a trio of FITS files
 *
 * Image data is written to (baseName)_img.fits
 * Variance is written to (baseName)_var.fits
 * Mask data is written to (baseName)_msk.fits
 */
template<typename ImagePixelT, typename MaskPixelT>
void lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::writeFits(std::string baseName) const {

    const std::string imageSuffix = "_img.fits";
    const std::string maskSuffix = "_msk.fits";
    const std::string varianceSuffix = "_var.fits";

    std::string fileName;

    fileName = baseName + imageSuffix;
    _imagePtr->writeFits(fileName);
    
    fileName = baseName + varianceSuffix;
    _variancePtr->writeFits(fileName);
    
    fileName = baseName + maskSuffix;
    _maskPtr->writeFits(fileName);
    
}


template<typename ImagePixelT, typename MaskPixelT>
typename lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::MaskedImagePtrT
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::getSubImage(const vw::BBox2i &subRegion) const {

    typename Image<ImagePixelT>::ImagePtrT newSubImage = _imagePtr->getSubImage(subRegion);
     
    typename Image<ImagePixelT>::ImagePtrT newSubVariance = _variancePtr->getSubImage(subRegion);

    typename Mask<MaskPixelT>::MaskPtrT newSubMask = _maskPtr->getSubMask(subRegion);

    typename MaskedImage<ImagePixelT, MaskPixelT>::MaskedImagePtrT newMaskedImage(
        new MaskedImage<ImagePixelT, MaskPixelT>(newSubImage, newSubVariance, newSubMask));

    return newMaskedImage; // and what happened to the metaData??
}

// Given a MaskedImage, insert insertImage, place it into this MaskedImage as directed by subRegion.
// An exception is generated if subRegion is not of the same size as insertImage.
//
template<typename ImagePixelT, typename MaskPixelT>
void lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::replaceSubImage(
    const vw::BBox2i &subRegion, MaskedImagePtrT insertImage, 
    const bool replaceMask, const bool replaceImage, const bool replaceVariance) {

    if (replaceImage) {
        typename Image<ImagePixelT>::ImageIVwT& _imageView = _imagePtr->getIVw();
        typename Image<ImagePixelT>::ImageIVwT& _imageViewInsert = insertImage->_imagePtr->getIVw();
        try {
            crop(_imageView, subRegion) = _imageViewInsert;
        } catch (std::exception eex) {
            throw lsst::pex::exceptions::Runtime(std::string("in ") + __func__);
        } 
    }
    if (replaceVariance) {
        typename Image<ImagePixelT>::ImageIVwT& _imageView = _variancePtr->getIVw();
        typename Image<ImagePixelT>::ImageIVwT& _imageViewInsert = insertImage->_variancePtr->getIVw();
        try {
            crop(_imageView, subRegion) = _imageViewInsert;
        } catch (std::exception eex) {
            throw lsst::pex::exceptions::Runtime(std::string("in ") + __func__);
        } 
    }
    if (replaceMask) {
        typename Mask<MaskPixelT>::MaskIVwT& _imageView = _maskPtr->getIVw();
        typename Mask<MaskPixelT>::MaskIVwT& _imageViewInsert = insertImage->_maskPtr->getIVw();
        try {
            crop(_imageView, subRegion) = _imageViewInsert;
        } catch (std::exception eex) {
            throw lsst::pex::exceptions::Runtime(std::string("in ") + __func__);
        } 
    }
}

// Set the pixel values of the variance based on the image.  The assumption is
// gaussian statistics, so that variance = image / k, where k is the gain in
// electrons per ADU

template<typename ImagePixelT, typename MaskPixelT>
void lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::setDefaultVariance() {
    double gain;

    try {
        gain = _imagePtr->getGain();
    } 

    catch (...) {
        lsst::pex::utils::Trace("afw.MaskedImage", 0,
            "Gain could not be set in setDefaultVariance().  Using gain=1.0");
        gain = 1.0;
    }

    typename Image<ImagePixelT>::ImageIVwPtrT imageVw = _imagePtr->getIVwPtr();
    typename Image<ImagePixelT>::ImageIVwPtrT varianceVw = _variancePtr->getIVwPtr();

    *varianceVw = *imageVw / gain;

    lsst::pex::utils::Trace("afw.MaskedImage", 1,
              boost::format("Using gain = %f in setDefaultVariance()") % gain);

}


template<typename ImagePixelT, typename MaskPixelT> 
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>&
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::operator+=(MaskedImage<ImagePixelT, MaskPixelT>& maskedImageInput) {
    *_maskPtr |= *(maskedImageInput.getMask());
    *_imagePtr += *(maskedImageInput.getImage());
    *_variancePtr += *(maskedImageInput.getVariance());
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>&
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::operator-=(MaskedImage<ImagePixelT, MaskPixelT>& maskedImageInput) {
    *_maskPtr |= *(maskedImageInput.getMask());
    *_imagePtr -= *(maskedImageInput.getImage());
    *_variancePtr += *(maskedImageInput.getVariance());
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>&
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::operator*=(MaskedImage<ImagePixelT, MaskPixelT>& maskedImageInput) {
    
    *_maskPtr |= *(maskedImageInput.getMask());
    *_imagePtr *= *(maskedImageInput.getImage());
    // For the variance arithmetic, reach down directly to the vw level:
    typename Image<ImagePixelT>::ImageIVwPtrT _imageVwPtr = _imagePtr->getIVwPtr();
    typename Image<ImagePixelT>::ImageIVwPtrT _varianceVwPtr = _variancePtr->getIVwPtr();
    typename Image<ImagePixelT>::ImageIVwPtrT _inpImageVwPtr = maskedImageInput.getImage()->getIVwPtr();
    typename Image<ImagePixelT>::ImageIVwPtrT _inpVarianceVwPtr = maskedImageInput.getVariance()->getIVwPtr();
    *_varianceVwPtr = (*_varianceVwPtr) * (*_inpImageVwPtr) + (*_inpVarianceVwPtr) * (*_imageVwPtr) + 
        (*_varianceVwPtr) * (*_inpVarianceVwPtr);
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>&
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::operator/=(MaskedImage<ImagePixelT, MaskPixelT>& maskedImageInput) {
    *_maskPtr |= *(maskedImageInput.getMask());
    *_imagePtr /= *(maskedImageInput.getImage());
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>&
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::operator+=(ImagePixelT scalar) {
    *_imagePtr += scalar;
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>&
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::operator-=(ImagePixelT scalar) {
    *_imagePtr -= scalar;
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>&
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::operator*=(ImagePixelT scalar) {
    *_imagePtr *= scalar;
    *_variancePtr *= scalar*scalar;
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>&
lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::operator/=(ImagePixelT scalar) {
    *_imagePtr /= scalar;
    *_variancePtr /= scalar*scalar;
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
void lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::processPixels(
    MaskPixelBooleanFunc<MaskPixelT> &selectionFunc, 
    PixelProcessingFunc<ImagePixelT, MaskPixelT> &processingFunc) {
    // Need to build ZipFilter iterator, call processingFunc()
}

template<typename ImagePixelT, typename MaskPixelT> 
void lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::processPixels(
    PixelProcessingFunc<ImagePixelT, MaskPixelT> &processingFunc
) {
    lsst::pex::utils::Trace("afw.MaskedImage", 5, "Processing pixels");

    PixelLocator<ImagePixelT> i = processingFunc.getImagePixelLocatorBegin();
    PixelLocator<ImagePixelT> iEnd = processingFunc.getImagePixelLocatorEnd();
    PixelLocator<MaskPixelT> m = processingFunc.getMaskPixelLocatorBegin();

    for ( ; i != iEnd; i++, m++) {
        processingFunc(i, m);
    }
}

template<typename ImagePixelT, typename MaskPixelT> 
void  lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::processPixels(
    MaskPixelBooleanFunc<MaskPixelT> &selectionFunc, 
    PixelProcessingFunc<ImagePixelT, MaskPixelT> &processingFunc,
    MaskedImage<ImagePixelT, MaskPixelT>&
) {
}

template<typename ImagePixelT, typename MaskPixelT> 
inline unsigned int  lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::getRows() const {
    return _imagePtr->getRows();
}

template<typename ImagePixelT, typename MaskPixelT> 
inline unsigned int  lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::getCols() const {
    return _imagePtr->getCols();
}

template<typename ImagePixelT, typename MaskPixelT> 
inline unsigned int  lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::getOffsetRows() const {
    return _imagePtr->getOffsetRows();
}

template<typename ImagePixelT, typename MaskPixelT> 
inline unsigned int  lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::getOffsetCols() const {
    return _imagePtr->getOffsetCols();
}

// private function conformSizes() ensures that the Mask and Variance have the same dimensions
// as Image.  If Mask and/or Variance have non-zero dimensions that conflict with the size of Image,
// an exception is thrown.

template<typename ImagePixelT, typename MaskPixelT>
void lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>::conformSizes() {

    typename Image<ImagePixelT>::ImageIVwPtrT _imageVwPtr = _imagePtr->getIVwPtr();
    typename Image<ImagePixelT>::ImageIVwPtrT _varianceVwPtr = _variancePtr->getIVwPtr();
    typename Mask<MaskPixelT>::MaskIVwPtrT _maskVwPtr = _maskPtr->getIVwPtr();

    unsigned int goldRows = _imageVwPtr->rows();
    unsigned int goldCols = _imageVwPtr->cols();

    unsigned int testRows, testCols;

    testRows = _varianceVwPtr->rows();
    testCols = _varianceVwPtr->cols();

    if (testRows > 0 && testRows != goldRows) throw;
    if (testCols > 0 && testCols != goldCols) throw;

    _varianceVwPtr->set_size(goldCols, goldRows);

    testRows = _maskVwPtr->rows();
    testCols = _maskVwPtr->cols();

    if (testRows > 0 && testRows != goldRows) throw;
    if (testCols > 0 && testCols != goldCols) throw;

    _maskVwPtr->set_size(goldCols, goldRows);

    _imageRows = goldRows;
    _imageCols = goldCols;

}

// Would be better to just declare the operator() to be virtual = 0 - but causes problems for Swig/Python

template<typename ImagePixelT, typename MaskPixelT>
void lsst::afw::PixelProcessingFunc<ImagePixelT, MaskPixelT>::operator() (ImageIteratorT& i, MaskIteratorT& m) {
    std::cout << "this should not happen!" << std::endl;
//     abort();
}

// construct with ImageView pointer to ensure smart pointer reference counting?

template<typename PixelT> 
lsst::afw::PixelLocator<PixelT>::PixelLocator(
    vw::ImageView<PixelT>* iv, vw::PixelIterator<vw::ImageView<PixelT> > ivIterator
) : 
    vw::PixelIterator<vw::ImageView<PixelT> > (ivIterator),
    _cols(iv->cols()),
    _rows(iv->rows()),
    _planes(iv->planes()),
    _cstride(1),
    _rstride(_cols),
    _pstride(_rows*_cols)
{
}

template<typename PixelT> 
lsst::afw::PixelLocator<PixelT>& lsst::afw::PixelLocator<PixelT>::advance(int dx, int dy) {
    int delta = dx*_cstride + dy*_rstride;
    *this += delta;
    return *this;
}

