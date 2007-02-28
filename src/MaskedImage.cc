// -*- lsst-c++ -*-
// Implementations of MaskedImage class methods
// This file can NOT be separately compiled!   It is included by MaskedImage.h

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>::MaskedImage() :
     _imagePtr(new Image<ImagePixelT>()),
     _maskPtr(new Mask<MaskPixelT>()),
     _image(*_imagePtr),
     _mask(*_maskPtr) {
}

template<typename ImagePixelT, typename MaskPixelT> 
MaskedImage<ImagePixelT, MaskPixelT>::MaskedImage(int nCols, int nRows) :
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
void  MaskedImage<ImagePixelT, MaskPixelT>::processPixels(MaskPixelBooleanFunc<MaskPixelT> selectionFunc, 
                                                          PixelProcessingFunc<ImagePixelT, MaskPixelT> processingFunc) {
    // Need to build ZipFilter iterator, call processingFunc()
}

template<typename ImagePixelT, typename MaskPixelT> 
void  MaskedImage<ImagePixelT, MaskPixelT>::processPixels(PixelProcessingFunc<ImagePixelT, MaskPixelT> processingFunc) {
    std::cout << "Processing pixels" << std::endl;
}

template<typename ImagePixelT, typename MaskPixelT> 
void  MaskedImage<ImagePixelT, MaskPixelT>::processPixels(MaskPixelBooleanFunc<MaskPixelT> selectionFunc, 
                                                          PixelProcessingFunc<ImagePixelT, MaskPixelT> processingFunc,
                                                          MaskedImage<ImagePixelT, MaskPixelT>&) {
}

// Would be better to just declare the operator() to be virtual = 0 - but causes problems for Swig/Python

template<typename ImagePixelT, typename MaskPixelT>
bool PixelProcessingFunc<ImagePixelT, MaskPixelT>::operator() (typename PixelProcessingFunc<ImagePixelT, MaskPixelT>::TupleT t) {
    abort();
    return true;
}
