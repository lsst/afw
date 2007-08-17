// -*- lsst-c++ -*-
// Implementations of MaskedImage class methods
// This file can NOT be separately compiled!   It is included by MaskedImage.h

#include <typeinfo>
#include "lsst/mwi/utils/Trace.h"
#include "lsst/mwi/exceptions/Exception.h"

using lsst::mwi::exceptions::NotFound;
namespace mwiu = lsst::mwi::utils;

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>::MaskedImage() :
    LsstBase(typeid(this)),
    _imagePtr(new Image<ImagePixelT>()),
    _variancePtr(new Image<ImagePixelT>()),
    _maskPtr(new Mask<MaskPixelT>()) {
}

// Construct from a supplied Image and Mask.  The Variance will be set to zero

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>::MaskedImage(ImagePtrT image, MaskPtrT mask) :
    LsstBase(typeid(this)),
    _imagePtr(image),
    _variancePtr(new Image<ImagePixelT>()),
    _maskPtr(mask) {
    conformSizes();
}    

// Construct from a supplied Image, Variance, and Mask.

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>::MaskedImage(ImagePtrT image, ImagePtrT variance, MaskPtrT mask) :
    LsstBase(typeid(this)),
    _imagePtr(image),
    _variancePtr(variance),
    _maskPtr(mask) {
    conformSizes();
}    


template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>::MaskedImage(int nCols, int nRows) :
    LsstBase(typeid(this)),
    _imagePtr(new Image<ImagePixelT>(nCols, nRows)),
    _variancePtr(new Image<ImagePixelT>(nCols, nRows)),
    _maskPtr(new Mask<MaskPixelT>(nCols, nRows)) {
}

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>::~MaskedImage() {
}

template<typename ImagePixelT, typename MaskPixelT>
boost::shared_ptr<Mask<MaskPixelT> > MaskedImage<ImagePixelT, MaskPixelT>::getMask() const{
    return _maskPtr;
}

template<typename ImagePixelT, typename MaskPixelT>
typename MaskedImage<ImagePixelT, MaskPixelT>::ImagePtrT MaskedImage<ImagePixelT, MaskPixelT>::getImage() const {
    return _imagePtr;
}

template<typename ImagePixelT, typename MaskPixelT>
typename MaskedImage<ImagePixelT, MaskPixelT>::ImagePtrT MaskedImage<ImagePixelT, MaskPixelT>::getVariance() const {
    return _variancePtr;
}

template<typename ImagePixelT, typename MaskPixelT>
void MaskedImage<ImagePixelT, MaskPixelT>::readFits(std::string baseName) {

    const std::string imageSuffix = "_img.fits";
    const std::string maskSuffix = "_msk.fits";
    const std::string varianceSuffix = "_var.fits";

    bool fileReadOK = false;

// reset any existing data

    typename Image<ImagePixelT>::ImageIVwPtrT _imageVwPtr = _imagePtr->getIVwPtr();
    typename Image<ImagePixelT>::ImageIVwPtrT _varianceVwPtr = _variancePtr->getIVwPtr();
    typename Mask<MaskPixelT>::MaskIVwPtrT _maskVwPtr = _maskPtr->getIVwPtr();

    _imageRows = 0;
    _imageCols = 0;
    _imageVwPtr->set_size(0,0);
    _varianceVwPtr->set_size(0,0);
    _maskVwPtr->set_size(0,0);

    std::string fileName;
    try {
        fileName = baseName + imageSuffix;
       _imagePtr->readFits(fileName);
       fileReadOK = true;
    }
    catch (vw::IOErr){
    }

    try {
        fileName = baseName + varianceSuffix;
       _variancePtr->readFits(fileName);
       fileReadOK = true;
    }
    catch (vw::IOErr){
    }

    try {
        fileName = baseName + maskSuffix;
        _maskPtr->readFits(fileName);
        fileReadOK = true;
     }
    catch (vw::IOErr) {
    }

//  if no file was read successfully, throw an exception

    if (fileReadOK == false) {
        throw NotFound(boost::format("Failed to open %s{%s,%s}") %
                       baseName % imageSuffix % maskSuffix);
    }

//  ensure all image components have the same size.  set_size is a nop if size would be unchanged

    conformSizes();

    Trace("fw.MaskedImage", 1,
              boost::format("Read in MaskedImage of size (%d,%d)") % _imageCols % _imageRows);

}

template<typename ImagePixelT, typename MaskPixelT>
void MaskedImage<ImagePixelT, MaskPixelT>::writeFits(std::string baseName) {

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
typename MaskedImage<ImagePixelT, MaskPixelT>::MaskedImagePtrT MaskedImage<ImagePixelT, MaskPixelT>::getSubImage(const vw::BBox2i subRegion) const {

    typename Image<ImagePixelT>::ImagePtrT newSubImage = _imagePtr->getSubImage(subRegion);
     
    typename Image<ImagePixelT>::ImagePtrT newSubVariance = _variancePtr->getSubImage(subRegion);

    typename Mask<MaskPixelT>::MaskPtrT newSubMask = _maskPtr->getSubMask(subRegion);

    typename MaskedImage<ImagePixelT, MaskPixelT>::MaskedImagePtrT newMaskedImage(new MaskedImage<ImagePixelT, MaskPixelT>(newSubImage, newSubVariance, newSubMask));

    return newMaskedImage; // and what happened to the metaData??
}

// Given a MaskedImage, insert insertImage, place it into this MaskedImage as directed by subRegion.
// An exception is generated if subRegion is not of the same size as insertImage.
//
template<typename ImagePixelT, typename MaskPixelT>
void MaskedImage<ImagePixelT, MaskPixelT>::replaceSubImage(const BBox2i subRegion, MaskedImagePtrT insertImage, 
                                         const bool replaceMask, const bool replaceImage, const bool replaceVariance)
{
    if (replaceImage) {
        typename Image<ImagePixelT>::ImageIVwT& _imageView = _imagePtr->getIVw();
        typename Image<ImagePixelT>::ImageIVwT& _imageViewInsert = insertImage->_imagePtr->getIVw();
        try {
            crop(_imageView, subRegion) = _imageViewInsert;
        } catch (exception eex) {
            throw Exception(std::string("in ") + __func__);
        } 
    }
    if (replaceVariance) {
        typename Image<ImagePixelT>::ImageIVwT& _imageView = _variancePtr->getIVw();
        typename Image<ImagePixelT>::ImageIVwT& _imageViewInsert = insertImage->_variancePtr->getIVw();
        try {
            crop(_imageView, subRegion) = _imageViewInsert;
        } catch (exception eex) {
            throw Exception(std::string("in ") + __func__);
        } 
    }
    if (replaceMask) {
        typename Mask<MaskPixelT>::MaskIVwT& _imageView = _maskPtr->getIVw();
        typename Mask<MaskPixelT>::MaskIVwT& _imageViewInsert = insertImage->_maskPtr->getIVw();
        try {
            crop(_imageView, subRegion) = _imageViewInsert;
        } catch (exception eex) {
            throw Exception(std::string("in ") + __func__);
        } 
    }
}

// Set the pixel values of the variance based on the image.  The assumption is
// gaussian statistics, so that variance = image / k, where k is the gain in
// electrons per ADU

template<typename ImagePixelT, typename MaskPixelT>
void MaskedImage<ImagePixelT, MaskPixelT>::setDefaultVariance()
{
    double gain;

    try {
        gain = _imagePtr->getGain();
    } 

    catch (...) {
        mwiu::Trace("fw.MaskedImage", 0, "Gain could not be set in setDefaultVariance().  Using gain=1.0");
        gain = 1.0;
    }

    typename Image<ImagePixelT>::ImageIVwPtrT imageVw = _imagePtr->getIVwPtr();
    typename Image<ImagePixelT>::ImageIVwPtrT varianceVw = _variancePtr->getIVwPtr();

    *varianceVw = *imageVw / gain;

    Trace("fw.MaskedImage", 1,
              boost::format("Using gain = %f in setDefaultVariance()") % gain);

}


template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>& MaskedImage<ImagePixelT, MaskPixelT>::operator+=(MaskedImageT & maskedImageInput) {
    *_maskPtr |= *(maskedImageInput.getMask());
    *_imagePtr += *(maskedImageInput.getImage());
    *_variancePtr += *(maskedImageInput.getVariance());
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>& MaskedImage<ImagePixelT, MaskPixelT>::operator-=(MaskedImageT & maskedImageInput) {
    *_maskPtr |= *(maskedImageInput.getMask());
    *_imagePtr -= *(maskedImageInput.getImage());
    *_variancePtr += *(maskedImageInput.getVariance());
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>& MaskedImage<ImagePixelT, MaskPixelT>::operator*=(MaskedImageT & maskedImageInput) {
    
    *_maskPtr |= *(maskedImageInput.getMask());
    *_imagePtr *= *(maskedImageInput.getImage());
    // For the variance arithmetic, reach down directly to the vw level:
    typename Image<ImagePixelT>::ImageIVwPtrT _imageIVwPtr = _imagePtr->getIVwPtr();
    typename Image<ImagePixelT>::ImageIVwPtrT _varianceIVwPtr = _variancePtr->getIVwPtr();
    typename Image<ImagePixelT>::ImageIVwPtrT _inpImageIVwPtr = maskedImageInput.getImage()->getIVwPtr();
    typename Image<ImagePixelT>::ImageIVwPtrT _inpVarianceIVwPtr = maskedImageInput.getVariance()->getIVwPtr();
    *_varianceIVwPtr = (*_varianceIVwPtr) * (*_inpImageIVwPtr) + (*_inpVarianceIVwPtr) * (*_imageIVwPtr) + 
        (*_varianceIVwPtr) * (*_inpVarianceIVwPtr);
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>& MaskedImage<ImagePixelT, MaskPixelT>::operator/=(MaskedImageT & maskedImageInput) {
    *_maskPtr |= *(maskedImageInput.getMask());
    *_imagePtr /= *(maskedImageInput.getImage());
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>& MaskedImage<ImagePixelT, MaskPixelT>::operator+=(ImagePixelT scalar) {
    *_imagePtr += scalar;
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>& MaskedImage<ImagePixelT, MaskPixelT>::operator-=(ImagePixelT scalar) {
    *_imagePtr -= scalar;
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>& MaskedImage<ImagePixelT, MaskPixelT>::operator*=(ImagePixelT scalar) {
    *_imagePtr *= scalar;
    *_variancePtr *= scalar*scalar;
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>& MaskedImage<ImagePixelT, MaskPixelT>::operator/=(ImagePixelT scalar) {
    *_imagePtr /= scalar;
    *_variancePtr /= scalar*scalar;
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
void  MaskedImage<ImagePixelT, MaskPixelT>::processPixels(MaskPixelBooleanFunc<MaskPixelT> &selectionFunc, 
                                                          PixelProcessingFunc<ImagePixelT, MaskPixelT> &processingFunc) {
    // Need to build ZipFilter iterator, call processingFunc()
}

template<typename ImagePixelT, typename MaskPixelT> 
void  MaskedImage<ImagePixelT, MaskPixelT>::processPixels(PixelProcessingFunc<ImagePixelT, MaskPixelT> &processingFunc) {
    mwiu::Trace("fw.MaskedImage", 1, "Processing pixels");

    PixelLocator<ImagePixelT> i = processingFunc.getImagePixelLocatorBegin();
    PixelLocator<ImagePixelT> iEnd = processingFunc.getImagePixelLocatorEnd();
    PixelLocator<MaskPixelT> m = processingFunc.getMaskPixelLocatorBegin();

    for ( ; i != iEnd; i++, m++) {
        processingFunc(i, m);
    }
}

template<typename ImagePixelT, typename MaskPixelT> 
void  MaskedImage<ImagePixelT, MaskPixelT>::processPixels(MaskPixelBooleanFunc<MaskPixelT> &selectionFunc, 
                                                          PixelProcessingFunc<ImagePixelT, MaskPixelT> &processingFunc,
                                                          MaskedImage<ImagePixelT, MaskPixelT>&) {
}

template<typename ImagePixelT, typename MaskPixelT> 
unsigned int  MaskedImage<ImagePixelT, MaskPixelT>::getRows() const
{
    return _imagePtr->getRows();
}

template<typename ImagePixelT, typename MaskPixelT> 
unsigned int  MaskedImage<ImagePixelT, MaskPixelT>::getCols() const
{
    return _imagePtr->getCols();
}

template<typename ImagePixelT, typename MaskPixelT> 
unsigned int  MaskedImage<ImagePixelT, MaskPixelT>::getOffsetRows() const
{
    return _imagePtr->getOffsetRows();
}

template<typename ImagePixelT, typename MaskPixelT> 
unsigned int  MaskedImage<ImagePixelT, MaskPixelT>::getOffsetCols() const
{
    return _imagePtr->getOffsetCols();
}

// private function conformSizes() ensures that the Mask and Variance have the same dimensions
// as Image.  If Mask and/or Variance have non-zero dimensions that conflict with the size of Image,
// an exception is thrown.

template<typename ImagePixelT, typename MaskPixelT>
void  MaskedImage<ImagePixelT, MaskPixelT>::conformSizes()
{

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
void PixelProcessingFunc<ImagePixelT, MaskPixelT>::operator() (ImageIteratorT& i, MaskIteratorT& m) {
    std::cout << "this should not happen!" << std::endl;
//     abort();
}

// construct with ImageView pointer to ensure smart pointer reference counting?

template<typename PixelT> 
PixelLocator<PixelT>::PixelLocator(vw::ImageView<PixelT>* iv, vw::PixelIterator<vw::ImageView<PixelT> > ivIterator) : 
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
PixelLocator<PixelT>& PixelLocator<PixelT>::advance(int dx, int dy) {
    int delta = dx*_cstride + dy*_rstride;
    *this += delta;
    return *this;
}

