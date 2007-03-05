// -*- lsst-c++ -*-
// Implementations of MaskedImage class methods
// This file can NOT be separately compiled!   It is included by MaskedImage.h

#include <typeinfo>

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
void  MaskedImage<ImagePixelT, MaskPixelT>::processPixels(MaskPixelBooleanFunc<MaskPixelT> &selectionFunc, 
                                                          PixelProcessingFunc<ImagePixelT, MaskPixelT> &processingFunc) {
    // Need to build ZipFilter iterator, call processingFunc()
}

template<typename ImagePixelT, typename MaskPixelT> 
void  MaskedImage<ImagePixelT, MaskPixelT>::processPixels(PixelProcessingFunc<ImagePixelT, MaskPixelT> &processingFunc) {
    std::cout << "Processing pixels" << std::endl;

    std::cout << typeid(processingFunc).name() << std::endl;

    ImageView<ImagePixelT>& vwImage = *(_image.getIVwPtr());
    ImageView<MaskPixelT>& vwMask = *(_mask.getIVwPtr());


// For unknown reasons this approach has compiler problems

// Just build zip iterator and iterate 
//     typename ImageView<ImagePixelT>::iterator beg1 = vwImage.begin();
//     typename ImageView<ImagePixelT>::iterator end1 = vwImage.end();

//     typename ImageView<MaskPixelT>::iterator beg2 = vwMask.begin();
//     typename ImageView<MaskPixelT>::iterator end2 = vwMask.end();

//     std::for_each(
//                   boost::make_zip_iterator(
//                                            boost::make_tuple(beg1, beg2)
//                                            ),
//                   boost::make_zip_iterator(
//                                            boost::make_tuple(end1, end2)
//                                            ),
//                   processingFunc()
//                   );

// So... use this somewhat less clean alternate approach 

    typedef boost::tuple<typename ImageView<ImagePixelT>::iterator, typename ImageView<MaskPixelT>::iterator> ZipTupleT;
    ZipTupleT zipBegin(vwImage.begin(), vwMask.begin());
    ZipTupleT zipEnd(vwImage.end(), vwMask.end());

    boost::zip_iterator<ZipTupleT> i(zipBegin);
    boost::zip_iterator<ZipTupleT> iEnd(zipEnd);

    for ( ; i != iEnd; i++) {
        processingFunc(*i);
    }
    
}

template<typename ImagePixelT, typename MaskPixelT> 
void  MaskedImage<ImagePixelT, MaskPixelT>::processPixels(MaskPixelBooleanFunc<MaskPixelT> &selectionFunc, 
                                                          PixelProcessingFunc<ImagePixelT, MaskPixelT> &processingFunc,
                                                          MaskedImage<ImagePixelT, MaskPixelT>&) {
}

// Would be better to just declare the operator() to be virtual = 0 - but causes problems for Swig/Python

template<typename ImagePixelT, typename MaskPixelT>
void PixelProcessingFunc<ImagePixelT, MaskPixelT>::operator() (boost::tuple<ImagePixelT&, MaskPixelT&> t) {
    std::cout << "this should not happen!" << std::endl;
//     abort();
}
