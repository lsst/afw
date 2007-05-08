// -*- lsst-c++ -*-
// Implementations of Image class methods
// This file can NOT be separately compiled!   It is included by Image.h

template<typename ImagePixelT>
Image<ImagePixelT>::Image() :
    LsstBase(typeid(this)),
    _imagePtr(new vw::ImageView<ImagePixelT>()),
    _image(*_imagePtr),
    _metaData(new DataProperty::DataProperty("FitsMetaData", 0)) {
}

template<typename ImagePixelT>
Image<ImagePixelT>::Image(int nCols, int nRows) :
    LsstBase(typeid(this)),
    _imagePtr(new vw::ImageView<ImagePixelT>(nCols, nRows)),
    _image(*_imagePtr),
    _metaData(new DataProperty::DataProperty("FitsMetaData", 0)) {
}

template<class ImagePixelT>
Image<ImagePixelT>::Image(ImageIVwPtrT image): 
    LsstBase(typeid(this)),
    _imagePtr(image),
    _image(*_imagePtr),
    _metaData(new DataProperty::DataProperty("FitsMetaData", 0)) {
}

template<class ImagePixelT> typename Image<ImagePixelT>::ImageIVwPtrT Image<ImagePixelT>::getIVwPtr() const 
{
    return _imagePtr;
}

template<class ImagePixelT> 
float Image<ImagePixelT>::getGain() const
{
    DataProperty::DataPropertyPtrT gainProp = _metaData->find("GAIN");
    if (gainProp) {
        std::string valueString = boost::any_cast<const std::string>(gainProp->getValue());
        float gain = atof(valueString.c_str());
        return gain;
    }
    throw Exception(std::string("in ") + __func__ + std::string(": Could not get gain from image metadata"));
}

template<class ImagePixelT>
void Image<ImagePixelT>::readFits(const string& fileName, int hdu)
{
    LSSTFitsResource<ImagePixelT> fitsRes;
    fitsRes.readFits(fileName, _image, _metaData, hdu);
}

template<class ImagePixelT>
void Image<ImagePixelT>::writeFits(const string& fileName)
{
    LSSTFitsResource<ImagePixelT> fitsRes;
    fitsRes.writeFits(_image, _metaData, fileName);
}

template<class ImagePixelT>
DataProperty::DataPropertyPtrT Image<ImagePixelT>::getMetaData()
{
    return _metaData;
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

template<class ImagePixelT> int Image<ImagePixelT>::getImageCols() const {
    return _image.cols();
}

template<class ImagePixelT> int Image<ImagePixelT>::getImageRows() const {
    return _image.rows();
}
