// -*- lsst-c++ -*-
// Implementations of MaskedImage class methods
// This file can NOT be separately compiled!   It is included by MaskedImage.h

#include <typeinfo>
#include <lsst/fw/Trace.h>

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>::MaskedImage() :
    LsstBase(typeid(this)),
    _imagePtr(new Image<ImagePixelT>()),
    _maskPtr(new Mask<MaskPixelT>()),
    _image(*_imagePtr),
    _mask(*_maskPtr) {
}

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>::MaskedImage(ImagePtrT image, MaskPtrT mask) :
    LsstBase(typeid(this)),
    _imagePtr(image),
    _maskPtr(mask),
    _image(*_imagePtr),
    _mask(*_maskPtr) {
}    

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>::MaskedImage(int nCols, int nRows) :
    LsstBase(typeid(this)),
    _imagePtr(new Image<ImagePixelT>(nCols, nRows)),
    _maskPtr(new Mask<MaskPixelT>(nCols, nRows)),
    _image(*_imagePtr),
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
void MaskedImage<ImagePixelT, MaskPixelT>::readFits(std::string baseName) {

    const std::string imageSuffix = "_img.fits";
    const std::string maskSuffix = "_msk.fits";
    const std::string varianceSuffix = "_var.fits";

// reset any existing data

    typename Image<ImagePixelT>::ImageIVwPtrT _imageVwPtr = _image.getIVwPtr();
    typename Mask<MaskPixelT>::MaskIVwPtrT _maskVwPtr = _mask.getIVwPtr();

    _imageRows = 0;
    _imageCols = 0;
    _imageVwPtr->set_size(0,0);
    _maskVwPtr->set_size(0,0);

    std::string fileName;
    try {
        fileName = baseName + imageSuffix;
       _imagePtr->readFits(fileName);
       _imageRows = _imageVwPtr->rows();
       _imageCols = _imageVwPtr->cols();
    }
    catch (vw::IOErr){
    }

    try {
        fileName = baseName + maskSuffix;
        _maskPtr->readFits(fileName);
        if (_imageRows > 0 && _maskVwPtr->rows() != (unsigned int)_imageRows) {
            throw;
        }
        if (_imageCols > 0 && _maskVwPtr->cols() != (unsigned int)_imageCols) {
            throw;
        }
       _imageRows = _maskVwPtr->rows();
       _imageCols = _maskVwPtr->cols();
     }
    catch (vw::IOErr) {
    }

//  if size not set now, no file was read successfully

    if (_imageRows==0) {
        throw;
    }

//  ensure all image components have the same size.  set_size is a nop if size would be unchanged

    _imageVwPtr->set_size(_imageCols, _imageRows);
    _maskVwPtr->set_size(_imageCols, _imageRows);

    fw::Trace("fw.MaskedImage", 1,
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
    
    fileName = baseName + maskSuffix;
    _maskPtr->writeFits(fileName);
    
}

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>& MaskedImage<ImagePixelT, MaskPixelT>::operator+=(MaskedImageT & maskedImageInput) {
    _mask |= *(maskedImageInput.getMask());
    _image += *(maskedImageInput.getImage());
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>& MaskedImage<ImagePixelT, MaskPixelT>::operator-=(MaskedImageT & maskedImageInput) {
    _mask |= *(maskedImageInput.getMask());
    _image -= *(maskedImageInput.getImage());
    return *this;
}

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>& MaskedImage<ImagePixelT, MaskPixelT>::operator*=(MaskedImageT & maskedImageInput) {
    _mask |= *(maskedImageInput.getMask());
    _image *= *(maskedImageInput.getImage());
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
