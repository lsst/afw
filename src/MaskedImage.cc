// -*- lsst-c++ -*-
// Implementations of MaskedImage class methods
// This file can NOT be separately compiled!   It is included by MaskedImage.h

#include <typeinfo>
#include "lsst/fw/Trace.h"
#include "lsst/fw/Exception.h"

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>::MaskedImage() :
    LsstBase(typeid(this)),
    _imagePtr(new Image<ImagePixelT>()),
    _variancePtr(new Image<ImagePixelT>()),
    _maskPtr(new Mask<MaskPixelT>()),
    _image(*_imagePtr),
    _variance(*_variancePtr),
    _mask(*_maskPtr) {
}

// Construct from a supplied Image and Mask.  The Variance will be set to zero

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>::MaskedImage(ImagePtrT image, MaskPtrT mask) :
    LsstBase(typeid(this)),
    _imagePtr(image),
    _variancePtr(new Image<ImagePixelT>()),
    _maskPtr(mask),
    _image(*_imagePtr),
    _variance(*_variancePtr),
    _mask(*_maskPtr) {
    conformSizes();
}    

// Construct from a supplied Image, Variance, and Mask.

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>::MaskedImage(ImagePtrT image, ImagePtrT variance, MaskPtrT mask) :
    LsstBase(typeid(this)),
    _imagePtr(image),
    _variancePtr(variance),
    _maskPtr(mask),
    _image(*_imagePtr),
    _variance(*_variancePtr),
    _mask(*_maskPtr) {
    conformSizes();
}    


template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>::MaskedImage(int nCols, int nRows) :
    LsstBase(typeid(this)),
    _imagePtr(new Image<ImagePixelT>(nCols, nRows)),
    _variancePtr(new Image<ImagePixelT>(nCols, nRows)),
    _maskPtr(new Mask<MaskPixelT>(nCols, nRows)),
    _image(*_imagePtr),
    _variance(*_imagePtr),
    _mask(*_maskPtr) {
}

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>::~MaskedImage() {
}

template<typename ImagePixelT, typename MaskPixelT>
boost::shared_ptr<Mask<MaskPixelT> > MaskedImage<ImagePixelT, MaskPixelT>::getMask() {
    return _maskPtr;
}

template<typename ImagePixelT, typename MaskPixelT>
typename MaskedImage<ImagePixelT, MaskPixelT>::ImagePtrT MaskedImage<ImagePixelT, MaskPixelT>::getImage() {
    return _imagePtr;
}

template<typename ImagePixelT, typename MaskPixelT>
typename MaskedImage<ImagePixelT, MaskPixelT>::ImagePtrT MaskedImage<ImagePixelT, MaskPixelT>::getVariance() {
    return _variancePtr;
}

template<typename ImagePixelT, typename MaskPixelT>
void MaskedImage<ImagePixelT, MaskPixelT>::readFits(std::string baseName) {

    const std::string imageSuffix = "_img.fits";
    const std::string maskSuffix = "_msk.fits";
    const std::string varianceSuffix = "_var.fits";

    bool fileReadOK = false;

// reset any existing data

    typename Image<ImagePixelT>::ImageIVwPtrT _imageVwPtr = _image.getIVwPtr();
    typename Image<ImagePixelT>::ImageIVwPtrT _varianceVwPtr = _image.getIVwPtr();
    typename Mask<MaskPixelT>::MaskIVwPtrT _maskVwPtr = _mask.getIVwPtr();

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
        throw lsst::fw::NotFound(boost::format("Failed to open %s{%s,%s}") %
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

// Set the pixel values of the variance based on the image.  The assumption is
// gaussian statistics, so that variance = image / k, where k is the gain in
// electrons per ADU

template<typename ImagePixelT, typename MaskPixelT>
void MaskedImage<ImagePixelT, MaskPixelT>::setDefaultVariance()
{
    float gain;

    try {
        gain = _image.getGain();
    } 

    catch (...) {
        fw::Trace("fw.MaskedImage", 0, "Gain could not be set in setDefaultVariance().  Using gain=1.0");
        gain = 1.0;
    }

    typename Image<ImagePixelT>::ImageIVwPtrT imageVw = _image.getIVwPtr();
    typename Image<ImagePixelT>::ImageIVwPtrT varianceVw = _variance.getIVwPtr();

    *varianceVw = *imageVw / gain;

    Trace("fw.MaskedImage", 1,
              boost::format("Using gain = %f in setDefaultVariance()") % gain);

}


template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>& MaskedImage<ImagePixelT, MaskPixelT>::operator+=(MaskedImageT & maskedImageInput) {
    _mask |= *(maskedImageInput.getMask());
    _image += *(maskedImageInput.getImage());
    _variance += *(maskedImageInput.getVariance());
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>& MaskedImage<ImagePixelT, MaskPixelT>::operator-=(MaskedImageT & maskedImageInput) {
    _mask |= *(maskedImageInput.getMask());
    _image -= *(maskedImageInput.getImage());
    _variance += *(maskedImageInput.getVariance());
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>& MaskedImage<ImagePixelT, MaskPixelT>::operator*=(MaskedImageT & maskedImageInput) {
    
    _mask |= *(maskedImageInput.getMask());
    _image *= *(maskedImageInput.getImage());
    // For the variance arithmetic, reach down directly to the vw level:
    typename Image<ImagePixelT>::ImageIVwPtrT _variancePtr = _variance.getIVwPtr();
    typename Image<ImagePixelT>::ImageIVwPtrT _imagePtr = _image.getIVwPtr();
    typename Image<ImagePixelT>::ImageIVwPtrT _inpImagePtr = maskedImageInput.getImage()->getIVwPtr();
    typename Image<ImagePixelT>::ImageIVwPtrT _inpVariancePtr = maskedImageInput.getImage()->getIVwPtr();
    *_variancePtr = (*_variancePtr) * (*_inpImagePtr) + (*_inpVariancePtr) * (*_imagePtr) + 
        (*_variancePtr) * (*_inpVariancePtr);
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>& MaskedImage<ImagePixelT, MaskPixelT>::operator/=(MaskedImageT & maskedImageInput) {
    _mask |= *(maskedImageInput.getMask());
    _image /= *(maskedImageInput.getImage());
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
void  MaskedImage<ImagePixelT, MaskPixelT>::processPixels(MaskPixelBooleanFunc<MaskPixelT> &selectionFunc, 
                                                          PixelProcessingFunc<ImagePixelT, MaskPixelT> &processingFunc) {
    // Need to build ZipFilter iterator, call processingFunc()
}

template<typename ImagePixelT, typename MaskPixelT> 
void  MaskedImage<ImagePixelT, MaskPixelT>::processPixels(PixelProcessingFunc<ImagePixelT, MaskPixelT> &processingFunc) {
    fw::Trace("fw.MaskedImage", 1, "Processing pixels");

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


// private function conformSizes() ensures that the Mask and Variance have the same dimensions
// as Image.  If Mask and/or Variance have non-zero dimensions that conflict with the size of Image,
// an exception is thrown.

template<typename ImagePixelT, typename MaskPixelT>
void  MaskedImage<ImagePixelT, MaskPixelT>::conformSizes()
{

    typename Image<ImagePixelT>::ImageIVwPtrT _imageVwPtr = _image.getIVwPtr();
    typename Image<ImagePixelT>::ImageIVwPtrT _varianceVwPtr = _variance.getIVwPtr();
    typename Mask<MaskPixelT>::MaskIVwPtrT _maskVwPtr = _mask.getIVwPtr();

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

