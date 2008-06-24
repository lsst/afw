// -*- lsst-c++ -*-
// Implementations of Image class methods

#include <stdexcept>

#include "boost/cstdint.hpp"

#include "lsst/afw/image/Image.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/LSSTFitsResource.h"

using namespace lsst::afw::image;

//
// Constructors
//
template<typename ImagePixelT>
Image<ImagePixelT>::Image() :
    lsst::daf::data::LsstBase(typeid(this)),
    _vwImagePtr(new vw::ImageView<ImagePixelT>()),
    _metaData(lsst::daf::base::DataProperty::createPropertyNode("FitsMetaData")),
    _offsetRows(0),
    _offsetCols(0)
{
}

template<typename ImagePixelT>
Image<ImagePixelT>::Image(int nCols, int nRows) :
    lsst::daf::data::LsstBase(typeid(this)),
    _vwImagePtr(new vw::ImageView<ImagePixelT>(nCols, nRows)),
    _metaData(lsst::daf::base::DataProperty::createPropertyNode("FitsMetaData")),
    _offsetRows(0),
    _offsetCols(0)
{
}

template<typename ImagePixelT>
Image<ImagePixelT>::Image(ImageIVwPtrT image): 
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
Image<ImagePixelT>& Image<ImagePixelT>::operator= (const Image<ImagePixelT>& image) {
    if (&image != this) {   // beware of self assignment: image = image;
        _vwImagePtr.reset();
        _vwImagePtr = image._vwImagePtr;
        _metaData = image._metaData;
        _offsetRows = image._offsetRows;
        _offsetCols = image._offsetCols;
    }
    
    return *this;
}

/**
 * @brief Return the gain from the image metadata
 *
 * @throw lsst::pex::exceptions::Runtime if gain not found
 */
template<typename ImagePixelT> 
double Image<ImagePixelT>::getGain() const {
    lsst::daf::base::DataProperty::PtrType gainProp = _metaData->findUnique("GAIN");
    if (gainProp) {
        double gain = boost::any_cast<const double>(gainProp->getValue());
        return gain;
    }
    throw lsst::pex::exceptions::Runtime(std::string("in ") + __func__ + std::string(": Could not get gain from image metadata"));
}

template<typename ImagePixelT>
void Image<ImagePixelT>::readFits(const std::string& fileName, int hdu) {
    LSSTFitsResource<ImagePixelT> fitsRes;
    fitsRes.readFits(fileName, *_vwImagePtr, _metaData, hdu);
}

template<typename ImagePixelT>
void Image<ImagePixelT>::writeFits(const std::string& fileName) const {
    LSSTFitsResource<ImagePixelT> fitsRes;
    fitsRes.writeFits(*_vwImagePtr, _metaData, fileName);
}

template<typename ImagePixelT>
lsst::daf::base::DataProperty::PtrType Image<ImagePixelT>::getMetaData() const {
    return _metaData;
}

template<typename ImagePixelT>
typename Image<ImagePixelT>::ImagePtrT
Image<ImagePixelT>::getSubImage(const vw::BBox2i imageRegion) const {


    // Check that imageRegion is completely inside the image
    
    vw::BBox2i imageBoundary(0, 0, getCols(), getRows());
    if (!imageBoundary.contains(imageRegion)) {
        throw lsst::pex::exceptions::InvalidParameter(boost::format("getSubImage region not contained within Image"));
    }

    ImageIVwPtrT croppedImage(new ImageIVwT());
    *croppedImage = copy(crop(*_vwImagePtr, imageRegion));
    ImagePtrT newImage(new Image<ImagePixelT>(croppedImage));
    vw::Vector<int, 2> bboxOffset = imageRegion.min();
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
 * @brief Given a Image, insertImage, place it into this Image as directed by maskRegion.
 *
 * @throw lsst::pex::exceptions::Runtime if maskRegion is not of the same size as insertImage.
 */
template<typename ImagePixelT>
void Image<ImagePixelT>::replaceSubImage(const vw::BBox2i maskRegion, ImagePtrT insertImage) {
    try {
        crop(*_vwImagePtr, maskRegion) = *(insertImage->getIVwPtr());
    } catch (std::exception eex) {
        throw lsst::pex::exceptions::Runtime(std::string("in ") + __func__);
    } 
}

template<typename ImagePixelT>
Image<ImagePixelT>&
Image<ImagePixelT>::operator += (const Image<ImagePixelT>& inputImage) {
    *_vwImagePtr += *(inputImage.getIVwPtr());
    return *this;
}

template<typename ImagePixelT>
Image<ImagePixelT>&
Image<ImagePixelT>::operator -= (const Image<ImagePixelT>& inputImage) {
    *_vwImagePtr -= *(inputImage.getIVwPtr());
    return *this;
}

template<typename ImagePixelT>
Image<ImagePixelT>& Image<ImagePixelT>::operator *= (const Image<ImagePixelT>& inputImage) {
    *_vwImagePtr *= *(inputImage.getIVwPtr());
    return *this;
}

template<typename ImagePixelT>
Image<ImagePixelT>& Image<ImagePixelT>::operator /= (const Image<ImagePixelT>& inputImage) {
    *_vwImagePtr /= *(inputImage.getIVwPtr());
    return *this;
}

template<typename ImagePixelT>
Image<ImagePixelT>& Image<ImagePixelT>::operator += (const ImagePixelT scalar) {
    *_vwImagePtr += scalar;
    return *this;
}

template<typename ImagePixelT>
Image<ImagePixelT>& Image<ImagePixelT>::operator -= (const ImagePixelT scalar) {
    *_vwImagePtr -= scalar;
    return *this;
}

template<typename ImagePixelT>
Image<ImagePixelT>& Image<ImagePixelT>::operator *= (const ImagePixelT scalar) {
    *_vwImagePtr *= scalar;
    return *this;
}

template<typename ImagePixelT>
Image<ImagePixelT>& Image<ImagePixelT>::operator /= (const ImagePixelT scalar) {
    *_vwImagePtr /= scalar;
    return *this;
}

/************************************************************************************************************/
//
// Explicit instantiations
//
template class Image<boost::uint16_t>;
template class Image<float>;
template class Image<double>;
