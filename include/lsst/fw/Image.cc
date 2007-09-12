// -*- lsst-c++ -*-
// Implementations of Image class methods
// This file can NOT be separately compiled!   It is included by Image.h
#include <stdexcept>
#include "lsst/mwi/data/SupportFactory.h"
#include <lsst/mwi/exceptions/Exception.h>

//
// Constructors
//

template<typename ImagePixelT>
lsst::fw::Image<ImagePixelT>::Image() :
    lsst::mwi::data::LsstBase(typeid(this)),
    _vwImagePtr(new vw::ImageView<ImagePixelT>()),
    _metaData(lsst::mwi::data::SupportFactory::createPropertyNode("FitsMetaData")),
    _offsetRows(0),
    _offsetCols(0)
{
}

template<typename ImagePixelT>
lsst::fw::Image<ImagePixelT>::Image(int nCols, int nRows) :
    lsst::mwi::data::LsstBase(typeid(this)),
    _vwImagePtr(new vw::ImageView<ImagePixelT>(nCols, nRows)),
    _metaData(lsst::mwi::data::SupportFactory::createPropertyNode("FitsMetaData")),
    _offsetRows(0),
    _offsetCols(0)
{
}

template<typename ImagePixelT>
lsst::fw::Image<ImagePixelT>::Image(ImageIVwPtrT image): 
    lsst::mwi::data::LsstBase(typeid(this)),
    _vwImagePtr(image),
    _metaData(lsst::mwi::data::SupportFactory::createPropertyNode("FitsMetaData")),
    _offsetRows(0),
    _offsetCols(0)
{
}

//
// Public Member Functions
//

template<typename ImagePixelT>
lsst::fw::Image<ImagePixelT>& lsst::fw::Image<ImagePixelT>::operator= (const Image<ImagePixelT>& image) {
    if (&image != this) {   // beware of self assignment: image = image;
        _vwImagePtr.reset();
        _vwImagePtr = image._vwImagePtr;
        _metaData = image._metaData;
        _offsetRows = image._offsetRows;
        _offsetCols = image._offsetCols;
    }
    
    return *this;
}

template<typename ImagePixelT>
typename lsst::fw::Image<ImagePixelT>::ImageIVwT& lsst::fw::Image<ImagePixelT>::getIVw() const {
    return *_vwImagePtr;
}


template<typename ImagePixelT>
typename lsst::fw::Image<ImagePixelT>::ImageIVwPtrT lsst::fw::Image<ImagePixelT>::getIVwPtr() const {
    return _vwImagePtr;
}

/**
 * \brief Return the gain from the image metadata
 *
 * \throw lsst::mwi::exceptions::Exception if gain not found
 */
template<typename ImagePixelT> 
double lsst::fw::Image<ImagePixelT>::getGain() const {
    lsst::mwi::data::DataProperty::PtrType gainProp = _metaData->findUnique("GAIN");
    if (gainProp) {
        double gain = boost::any_cast<const double>(gainProp->getValue());
        return gain;
    }
    throw lsst::mwi::exceptions::Exception(std::string("in ") + __func__ + std::string(": Could not get gain from image metadata"));
}

template<typename ImagePixelT>
void lsst::fw::Image<ImagePixelT>::readFits(const std::string& fileName, int hdu) {
    LSSTFitsResource<ImagePixelT> fitsRes;
    fitsRes.readFits(fileName, *_vwImagePtr, _metaData, hdu);
}

template<typename ImagePixelT>
void lsst::fw::Image<ImagePixelT>::writeFits(const std::string& fileName) const {
    LSSTFitsResource<ImagePixelT> fitsRes;
    fitsRes.writeFits(*_vwImagePtr, _metaData, fileName);
}

template<typename ImagePixelT>
lsst::mwi::data::DataProperty::PtrType lsst::fw::Image<ImagePixelT>::getMetaData() const {
    return _metaData;
}

template<typename ImagePixelT>
typename lsst::fw::Image<ImagePixelT>::ImagePtrT
lsst::fw::Image<ImagePixelT>::getSubImage(const BBox2i imageRegion) const {


    // Check that imageRegion is completely inside the image
    
    BBox2i imageBoundary(0, 0, getCols(), getRows());
    if (!imageBoundary.contains(imageRegion)) {
        throw lsst::mwi::exceptions::InvalidParameter(boost::format("getSubImage region not contained within Image"));
    }

    ImageIVwPtrT croppedImage(new ImageIVwT());
    *croppedImage = copy(crop(*_vwImagePtr, imageRegion));
    ImagePtrT newImage(new Image<ImagePixelT>(croppedImage));
    Vector<int, 2> bboxOffset = imageRegion.min();
    newImage->setOffsetRows(bboxOffset[1] + _offsetRows);
    newImage->setOffsetCols(bboxOffset[0] + _offsetCols);

    return newImage;
}

/**
 * \brief Given a Image, insertImage, place it into this Image as directed by maskRegion.
 *
 * \throw lsst::mwi::exceptions::Exception if maskRegion is not of the same size as insertImage.
 */
template<typename ImagePixelT>
void lsst::fw::Image<ImagePixelT>::replaceSubImage(const BBox2i maskRegion, ImagePtrT insertImage) {
    try {
        crop(*_vwImagePtr, maskRegion) = *(insertImage->getIVwPtr());
    } catch (std::exception eex) {
        throw lsst::mwi::exceptions::Exception(std::string("in ") + __func__);
    } 
}

template<typename ImagePixelT>
inline typename lsst::fw::Image<ImagePixelT>::ImageChannelT
lsst::fw::Image<ImagePixelT>::operator ()(int x, int y) const {
     return (*_vwImagePtr)(x, y);
}

template<typename ImagePixelT>
inline typename lsst::fw::Image<ImagePixelT>::pixel_accessor lsst::fw::Image<ImagePixelT>::origin() const {
    return getIVwPtr()->origin();
}

template<typename ImagePixelT>
lsst::fw::Image<ImagePixelT>&
lsst::fw::Image<ImagePixelT>::operator += (const Image<ImagePixelT>& inputImage) {
    *_vwImagePtr += *(inputImage.getIVwPtr());
    return *this;
}

template<typename ImagePixelT>
lsst::fw::Image<ImagePixelT>&
lsst::fw::Image<ImagePixelT>::operator -= (const Image<ImagePixelT>& inputImage) {
    *_vwImagePtr -= *(inputImage.getIVwPtr());
    return *this;
}

template<typename ImagePixelT>
inline lsst::fw::Image<ImagePixelT>&
lsst::fw::Image<ImagePixelT>::operator *= (const Image<ImagePixelT>& inputImage) {
    *_vwImagePtr *= *(inputImage.getIVwPtr());
    return *this;
}

template<typename ImagePixelT>
lsst::fw::Image<ImagePixelT>&
lsst::fw::Image<ImagePixelT>::operator /= (const Image<ImagePixelT>& inputImage) {
    *_vwImagePtr /= *(inputImage.getIVwPtr());
    return *this;
}

template<typename ImagePixelT>
lsst::fw::Image<ImagePixelT>& lsst::fw::Image<ImagePixelT>::operator += (const ImagePixelT scalar) {
    *_vwImagePtr += scalar;
    return *this;
}

template<typename ImagePixelT>
lsst::fw::Image<ImagePixelT>& lsst::fw::Image<ImagePixelT>::operator -= (const ImagePixelT scalar) {
    *_vwImagePtr -= scalar;
    return *this;
}

template<typename ImagePixelT>
lsst::fw::Image<ImagePixelT>& lsst::fw::Image<ImagePixelT>::operator *= (const ImagePixelT scalar) {
    *_vwImagePtr *= scalar;
    return *this;
}

template<typename ImagePixelT>
lsst::fw::Image<ImagePixelT>& lsst::fw::Image<ImagePixelT>::operator /= (const ImagePixelT scalar) {
    *_vwImagePtr /= scalar;
    return *this;
}

template<typename ImagePixelT>
inline unsigned int lsst::fw::Image<ImagePixelT>::getCols() const {
    return _vwImagePtr->cols();
}

template<typename ImagePixelT>
inline unsigned int lsst::fw::Image<ImagePixelT>::getRows() const {
    return _vwImagePtr->rows();
}

template<typename ImagePixelT>
inline unsigned int lsst::fw::Image<ImagePixelT>::getOffsetCols() const {
    return _offsetCols;
}

template<typename ImagePixelT>
inline unsigned int lsst::fw::Image<ImagePixelT>::getOffsetRows() const {
    return _offsetRows;
}

//
// Private Member Functions
//

template<typename ImagePixelT>
inline void lsst::fw::Image<ImagePixelT>::setOffsetRows(unsigned int offset) {
    _offsetRows = offset;
}

template<typename ImagePixelT>
inline void lsst::fw::Image<ImagePixelT>::setOffsetCols(unsigned int offset) {
    _offsetCols = offset;
}
