// -*- lsst-c++ -*-
// Implementations of Image class methods
// This file can NOT be separately compiled!   It is included by Image.h
#include <stdexcept>

#include <lsst/daf/base.h>
#include <lsst/pex/exceptions.h>
#include <lsst/afw/image/LSSTFitsResource.h>

//
// Constructors
//

template<typename ImagePixelT>
lsst::afw::image::Image<ImagePixelT>::Image() :
    lsst::daf::data::LsstBase(typeid(this)),
    _vwImagePtr(new vw::ImageView<ImagePixelT>()),
    _metaData(lsst::daf::base::DataProperty::createPropertyNode("FitsMetaData")),
    _offsetRows(0),
    _offsetCols(0)
{
}

template<typename ImagePixelT>
lsst::afw::image::Image<ImagePixelT>::Image(int nCols, int nRows) :
    lsst::daf::data::LsstBase(typeid(this)),
    _vwImagePtr(new vw::ImageView<ImagePixelT>(nCols, nRows)),
    _metaData(lsst::daf::base::DataProperty::createPropertyNode("FitsMetaData")),
    _offsetRows(0),
    _offsetCols(0)
{
}

template<typename ImagePixelT>
lsst::afw::image::Image<ImagePixelT>::Image(ImageIVwPtrT image): 
    lsst::daf::data::LsstBase(typeid(this)),
    _vwImagePtr(image),
    _metaData(lsst::daf::base::DataProperty::createPropertyNode("FitsMetaData")),
    _offsetRows(0),
    _offsetCols(0)
{
}

//
// Public Member Functions
//

template<typename ImagePixelT>
lsst::afw::image::Image<ImagePixelT>& lsst::afw::image::Image<ImagePixelT>::operator= (const Image<ImagePixelT>& image) {
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
typename lsst::afw::image::Image<ImagePixelT>::ImageIVwT& lsst::afw::image::Image<ImagePixelT>::getIVw() const {
    return *_vwImagePtr;
}


template<typename ImagePixelT>
typename lsst::afw::image::Image<ImagePixelT>::ImageIVwPtrT lsst::afw::image::Image<ImagePixelT>::getIVwPtr() const {
    return _vwImagePtr;
}

/**
 * \brief Return the gain from the image metadata
 *
 * \throw lsst::pex::exceptions::Runtime if gain not found
 */
template<typename ImagePixelT> 
double lsst::afw::image::Image<ImagePixelT>::getGain() const {
    lsst::daf::base::DataProperty::PtrType gainProp = _metaData->findUnique("GAIN");
    if (gainProp) {
        double gain = boost::any_cast<const double>(gainProp->getValue());
        return gain;
    }
    throw lsst::pex::exceptions::Runtime(std::string("in ") + __func__ + std::string(": Could not get gain from image metadata"));
}

template<typename ImagePixelT>
void lsst::afw::image::Image<ImagePixelT>::readFits(const std::string& fileName, int hdu) {
    lsst::afw::image::LSSTFitsResource<ImagePixelT> fitsRes;
    fitsRes.readFits(fileName, *_vwImagePtr, _metaData, hdu);
}

template<typename ImagePixelT>
void lsst::afw::image::Image<ImagePixelT>::writeFits(const std::string& fileName) const {
    lsst::afw::image::LSSTFitsResource<ImagePixelT> fitsRes;
    fitsRes.writeFits(*_vwImagePtr, _metaData, fileName);
}

template<typename ImagePixelT>
lsst::daf::base::DataProperty::PtrType lsst::afw::image::Image<ImagePixelT>::getMetaData() const {
    return _metaData;
}

template<typename ImagePixelT>
typename lsst::afw::image::Image<ImagePixelT>::ImagePtrT
lsst::afw::image::Image<ImagePixelT>::getSubImage(const vw::BBox2i imageRegion) const {


    // Check that imageRegion is completely inside the image
    
    BBox2i imageBoundary(0, 0, getCols(), getRows());
    if (!imageBoundary.contains(imageRegion)) {
        throw lsst::pex::exceptions::InvalidParameter(boost::format("getSubImage region not contained within Image"));
    }

    ImageIVwPtrT croppedImage(new ImageIVwT());
    *croppedImage = copy(crop(*_vwImagePtr, imageRegion));
    ImagePtrT newImage(new Image<ImagePixelT>(croppedImage));
    Vector<int, 2> bboxOffset = imageRegion.min();
    newImage->setOffsetRows(bboxOffset[1] + _offsetRows);
    newImage->setOffsetCols(bboxOffset[0] + _offsetCols);

    // Make a copy of the metadata

    lsst::daf::base::DataProperty::PtrType newMetaData(new lsst::daf::base::DataProperty(*_metaData));
    newImage->_metaData = newMetaData;

    // If CRPIX values are present in _metaData, keep them consistent with the offset

    lsst::daf::base::DataProperty::PtrType crpix1 = newImage->_metaData->findUnique("CRPIX1");
    if (crpix1) {
        double crpix1Value = boost::any_cast<double>(crpix1->getValue());
        crpix1->setValue(crpix1Value - bboxOffset[0]);
    }

    lsst::daf::base::DataProperty::PtrType crpix2 = newImage->_metaData->findUnique("CRPIX2");
    if (crpix2) {
        double crpix2Value = boost::any_cast<double>(crpix2->getValue());
        crpix2->setValue(crpix2Value - bboxOffset[1]);
    }

    return newImage;
}

/**
 * \brief Given a Image, insertImage, place it into this Image as directed by maskRegion.
 *
 * \throw lsst::pex::exceptions::Runtime if maskRegion is not of the same size as insertImage.
 */
template<typename ImagePixelT>
void lsst::afw::image::Image<ImagePixelT>::replaceSubImage(const vw::BBox2i maskRegion, ImagePtrT insertImage) {
    try {
        crop(*_vwImagePtr, maskRegion) = *(insertImage->getIVwPtr());
    } catch (std::exception eex) {
        throw lsst::pex::exceptions::Runtime(std::string("in ") + __func__);
    } 
}

template<typename ImagePixelT>
inline typename lsst::afw::image::Image<ImagePixelT>::ImageChannelT
lsst::afw::image::Image<ImagePixelT>::operator ()(int x, int y) const {
     return (*_vwImagePtr)(x, y);
}

template<typename ImagePixelT>
inline typename lsst::afw::image::Image<ImagePixelT>::pixel_accessor lsst::afw::image::Image<ImagePixelT>::origin() const {
    return getIVwPtr()->origin();
}

template<typename ImagePixelT>
lsst::afw::image::Image<ImagePixelT>&
lsst::afw::image::Image<ImagePixelT>::operator += (const Image<ImagePixelT>& inputImage) {
    *_vwImagePtr += *(inputImage.getIVwPtr());
    return *this;
}

template<typename ImagePixelT>
lsst::afw::image::Image<ImagePixelT>&
lsst::afw::image::Image<ImagePixelT>::operator -= (const Image<ImagePixelT>& inputImage) {
    *_vwImagePtr -= *(inputImage.getIVwPtr());
    return *this;
}

template<typename ImagePixelT>
inline lsst::afw::image::Image<ImagePixelT>&
lsst::afw::image::Image<ImagePixelT>::operator *= (const Image<ImagePixelT>& inputImage) {
    *_vwImagePtr *= *(inputImage.getIVwPtr());
    return *this;
}

template<typename ImagePixelT>
lsst::afw::image::Image<ImagePixelT>&
lsst::afw::image::Image<ImagePixelT>::operator /= (const Image<ImagePixelT>& inputImage) {
    *_vwImagePtr /= *(inputImage.getIVwPtr());
    return *this;
}

template<typename ImagePixelT>
lsst::afw::image::Image<ImagePixelT>& lsst::afw::image::Image<ImagePixelT>::operator += (const ImagePixelT scalar) {
    *_vwImagePtr += scalar;
    return *this;
}

template<typename ImagePixelT>
lsst::afw::image::Image<ImagePixelT>& lsst::afw::image::Image<ImagePixelT>::operator -= (const ImagePixelT scalar) {
    *_vwImagePtr -= scalar;
    return *this;
}

template<typename ImagePixelT>
lsst::afw::image::Image<ImagePixelT>& lsst::afw::image::Image<ImagePixelT>::operator *= (const ImagePixelT scalar) {
    *_vwImagePtr *= scalar;
    return *this;
}

template<typename ImagePixelT>
lsst::afw::image::Image<ImagePixelT>& lsst::afw::image::Image<ImagePixelT>::operator /= (const ImagePixelT scalar) {
    *_vwImagePtr /= scalar;
    return *this;
}

template<typename ImagePixelT>
inline unsigned int lsst::afw::image::Image<ImagePixelT>::getCols() const {
    return _vwImagePtr->cols();
}

template<typename ImagePixelT>
inline unsigned int lsst::afw::image::Image<ImagePixelT>::getRows() const {
    return _vwImagePtr->rows();
}

template<typename ImagePixelT>
inline unsigned int lsst::afw::image::Image<ImagePixelT>::getOffsetCols() const {
    return _offsetCols;
}

template<typename ImagePixelT>
inline unsigned int lsst::afw::image::Image<ImagePixelT>::getOffsetRows() const {
    return _offsetRows;
}

//
// Private Member Functions
//

template<typename ImagePixelT>
inline void lsst::afw::image::Image<ImagePixelT>::setOffsetRows(unsigned int offset) {
    _offsetRows = offset;
}

template<typename ImagePixelT>
inline void lsst::afw::image::Image<ImagePixelT>::setOffsetCols(unsigned int offset) {
    _offsetCols = offset;
}
