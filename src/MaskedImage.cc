// -*- lsst-c++ -*-
// Implementations of MaskedImage class methods
// This file can NOT be separately compiled!   It is included by MaskedImage.h

#include <typeinfo>

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
boost::shared_ptr<Image<ImagePixelT> > MaskedImage<ImagePixelT, MaskPixelT>::getImage() {
    return _imagePtr;
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
    std::cout << "Processing pixels" << std::endl;

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
