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

template<class ImagePixelT> typename Image<ImagePixelT>::ImageIVwT& Image<ImagePixelT>::getIVw() const {
    return _image;
}


template<class ImagePixelT> typename Image<ImagePixelT>::ImageIVwPtrT Image<ImagePixelT>::getIVwPtr() const 
{
    return _imagePtr;
}

template<class ImagePixelT> 
double Image<ImagePixelT>::getGain() const
{
    DataPropertyPtrT gainProp = _metaData->find("GAIN");
    if (gainProp) {
        double gain = boost::any_cast<const double>(gainProp->getValue());
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
DataPropertyPtrT Image<ImagePixelT>::getMetaData()
{
    return _metaData;
}

template<class ImagePixelT>
typename Image<ImagePixelT>::ImagePtrT Image<ImagePixelT>::getSubImage(const vw::BBox2i imageRegion) const {

    ImageIVwPtrT croppedImage(new ImageIVwT());
    *croppedImage = copy(crop(_image, imageRegion));
    ImagePtrT newImage(new Image<ImagePixelT>(croppedImage));
    return newImage;
}

// Given a Image, insertImage, place it into this Image as directed by maskRegion.
// An exception is generated if maskRegion is not of the same size as insertImage.
//
template<class ImagePixelT>
void Image<ImagePixelT>::replaceSubImage(const BBox2i maskRegion, ImagePtrT insertImage)
{
    try {
        crop(_image, maskRegion) = insertImage->_image;
    } catch (exception eex) {
        throw Exception(std::string("in ") + __func__);
    } 
}

template<class ImagePixelT> typename Image<ImagePixelT>::ImageChannelT Image<ImagePixelT>::operator ()(int x, int y) const
{
//      cout << x << " " << y << " " << (void *)_image(x, y) << endl;
     return _image(x, y);
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

template<class ImagePixelT> Image<ImagePixelT>&  Image<ImagePixelT>::operator += (const ImagePixelT scalar)
{
    _image += scalar;
    return *this;
}

template<class ImagePixelT> Image<ImagePixelT>&  Image<ImagePixelT>::operator -= (const ImagePixelT scalar)
{
    _image -= scalar;
    return *this;
}

template<class ImagePixelT> Image<ImagePixelT>&  Image<ImagePixelT>::operator *= (const ImagePixelT scalar)
{
    _image *= scalar;
    return *this;
}

template<class ImagePixelT> Image<ImagePixelT>&  Image<ImagePixelT>::operator /= (const ImagePixelT scalar)
{
    _image /= scalar;
    return *this;
}

template<class ImagePixelT> int Image<ImagePixelT>::getImageCols() const {
    return _image.cols();
}

template<class ImagePixelT> int Image<ImagePixelT>::getImageRows() const {
    return _image.rows();
}
