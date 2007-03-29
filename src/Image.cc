// -*- lsst-c++ -*-
// Implementations of Image class methods
// This file can NOT be separately compiled!   It is included by Image.h

template<typename ImagePixelT>
Image<ImagePixelT>::Image() :
    _imagePtr(new vw::ImageView<ImagePixelT>()),
    _image(*_imagePtr) {
}

template<typename ImagePixelT>
Image<ImagePixelT>::Image(int nCols, int nRows) :
    _imagePtr(new vw::ImageView<ImagePixelT>(nCols, nRows)),
    _image(*_imagePtr) {
}

template<class ImagePixelT>
Image<ImagePixelT>::Image(ImageIVwPtrT image): 
    _imagePtr(image),
    _image(*_imagePtr) {
    _imageRows = _image.rows();
    _imageCols = _image.cols();

}

template<class ImagePixelT> typename Image<ImagePixelT>::ImageIVwPtrT Image<ImagePixelT>::getIVwPtr() const {
    return _imagePtr;
}

template<class ImagePixelT> Image<ImagePixelT>&  Image<ImagePixelT>::operator += (const Image<ImagePixelT>& inputImage)
{
    _image += *(inputImage.getIVwPtr());
    return *this;
}
