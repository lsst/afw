// -*- lsst-c++ -*-
// Implementations of Image class methods
// This file can NOT be separately compiled!   It is included by Image.h

template<typename ImagePixelT>
Image<ImagePixelT>::Image() :
    fw::LsstBase(typeid(this)),
    _imagePtr(new vw::ImageView<ImagePixelT>()),
    _image(*_imagePtr),
    _metaData(new DataProperty::DataProperty("FitsMetaData", 0)) {
}

template<typename ImagePixelT>
Image<ImagePixelT>::Image(int nCols, int nRows) :
    fw::LsstBase(typeid(this)),
    _imagePtr(new vw::ImageView<ImagePixelT>(nCols, nRows)),
    _image(*_imagePtr),
    _metaData(new DataProperty::DataProperty("FitsMetaData", 0)) {
}

template<class ImagePixelT>
Image<ImagePixelT>::Image(ImageIVwPtrT image): 
    fw::LsstBase(typeid(this)),
    _imagePtr(image),
    _image(*_imagePtr),
    _metaData(new DataProperty::DataProperty("FitsMetaData", 0)) {
    _imageRows = _image.rows();
    _imageCols = _image.cols();

}

template<class ImagePixelT> typename Image<ImagePixelT>::ImageIVwPtrT Image<ImagePixelT>::getIVwPtr() const {
    return _imagePtr;
}

template<class ImagePixelT>
void Image<ImagePixelT>::readFits(const string& fileName, int hdu)
{
    lsst::LSSTFitsResource<ImagePixelT> fitsRes;
    fitsRes.readFits(fileName, _image, _metaData, hdu);
}

template<class ImagePixelT>
void Image<ImagePixelT>::writeFits(const string& fileName)
{
    lsst::LSSTFitsResource<ImagePixelT> fitsRes;
    fitsRes.writeFits(_image, _metaData, fileName);
}

template<class ImagePixelT> Image<ImagePixelT>&  Image<ImagePixelT>::operator += (const Image<ImagePixelT>& inputImage)
{
    _image += *(inputImage.getIVwPtr());
    return *this;
}

template<class ImagePixelT> Image<ImagePixelT>&  Image<ImagePixelT>::operator -= (const Image<ImagePixelT>& inputImage)
{
    _image -= *(inputImage.getIVwPtr());
    return *this;
}

template<class ImagePixelT> Image<ImagePixelT>&  Image<ImagePixelT>::operator *= (const Image<ImagePixelT>& inputImage)
{
    _image *= *(inputImage.getIVwPtr());
    return *this;
}

template<class ImagePixelT> Image<ImagePixelT>&  Image<ImagePixelT>::operator /= (const Image<ImagePixelT>& inputImage)
{
    _image /= *(inputImage.getIVwPtr());
    return *this;
}

