// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008-2016  AURA/LSST.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

/*
 * Implementation for MaskedImage
 */
#include <cstdint>

#include "boost/format.hpp"
#include "lsst/log/Log.h"
#include "lsst/pex/exceptions.h"

#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/fits.h"
#include "lsst/afw/image/MaskedImageFitsReader.h"

namespace lsst {
namespace afw {
namespace image {

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::MaskedImage(unsigned int width, unsigned int height,
                                                                  MaskPlaneDict const& planeDict)
        : _image(new Image(width, height)),
          _mask(new Mask(width, height, planeDict)),
          _variance(new Variance(width, height)) {
    *_image = 0;
    *_mask = 0x0;
    *_variance = 0;
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::MaskedImage(lsst::geom::Extent2I const& dimensions,
                                                                  MaskPlaneDict const& planeDict)
        : _image(new Image(dimensions)),
          _mask(new Mask(dimensions, planeDict)),
          _variance(new Variance(dimensions)) {
    *_image = 0;
    *_mask = 0x0;
    *_variance = 0;
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::MaskedImage(lsst::geom::Box2I const& bbox,
                                                                  MaskPlaneDict const& planeDict)
        : _image(new Image(bbox)), _mask(new Mask(bbox, planeDict)), _variance(new Variance(bbox)) {
    *_image = 0;
    *_mask = 0x0;
    *_variance = 0;
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::MaskedImage(
        std::string const& fileName, std::shared_ptr<daf::base::PropertySet> metadata,
        lsst::geom::Box2I const& bbox, ImageOrigin origin, bool conformMasks, bool needAllHdus,
        std::shared_ptr<daf::base::PropertySet> imageMetadata,
        std::shared_ptr<daf::base::PropertySet> maskMetadata,
        std::shared_ptr<daf::base::PropertySet> varianceMetadata, bool allowUnsafe)
        : _image(), _mask(), _variance() {
    MaskedImageFitsReader reader(fileName);
    *this = reader.read<ImagePixelT, MaskPixelT, VariancePixelT>(bbox, origin, conformMasks, needAllHdus,
                                                                 allowUnsafe);
    if (metadata) {
        metadata->combine(*reader.readPrimaryMetadata());
    }
    if (imageMetadata) {
        imageMetadata->combine(*reader.readImageMetadata());
    }
    if (maskMetadata) {
        maskMetadata->combine(*reader.readMaskMetadata());
    }
    if (varianceMetadata) {
        varianceMetadata->combine(*reader.readVarianceMetadata());
    }
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::MaskedImage(
        fits::MemFileManager& manager, std::shared_ptr<daf::base::PropertySet> metadata,
        lsst::geom::Box2I const& bbox, ImageOrigin origin, bool conformMasks, bool needAllHdus,
        std::shared_ptr<daf::base::PropertySet> imageMetadata,
        std::shared_ptr<daf::base::PropertySet> maskMetadata,
        std::shared_ptr<daf::base::PropertySet> varianceMetadata, bool allowUnsafe)
        : _image(), _mask(), _variance() {
    MaskedImageFitsReader reader(manager);
    *this = reader.read<ImagePixelT, MaskPixelT, VariancePixelT>(bbox, origin, conformMasks, needAllHdus,
                                                                 allowUnsafe);
    if (metadata) {
        metadata->combine(*reader.readPrimaryMetadata());
    }
    if (imageMetadata) {
        imageMetadata->combine(*reader.readImageMetadata());
    }
    if (maskMetadata) {
        maskMetadata->combine(*reader.readMaskMetadata());
    }
    if (varianceMetadata) {
        varianceMetadata->combine(*reader.readVarianceMetadata());
    }
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::MaskedImage(
        fits::Fits& fitsFile, std::shared_ptr<daf::base::PropertySet> metadata, lsst::geom::Box2I const& bbox,
        ImageOrigin origin, bool conformMasks, bool needAllHdus,
        std::shared_ptr<daf::base::PropertySet> imageMetadata,
        std::shared_ptr<daf::base::PropertySet> maskMetadata,
        std::shared_ptr<daf::base::PropertySet> varianceMetadata, bool allowUnsafe)
        : _image(), _mask(), _variance() {
    MaskedImageFitsReader reader(&fitsFile);
    *this = reader.read<ImagePixelT, MaskPixelT, VariancePixelT>(bbox, origin, conformMasks, needAllHdus,
                                                                 allowUnsafe);
    if (metadata) {
        metadata->combine(*reader.readPrimaryMetadata());
    }
    if (imageMetadata) {
        imageMetadata->combine(*reader.readImageMetadata());
    }
    if (maskMetadata) {
        maskMetadata->combine(*reader.readMaskMetadata());
    }
    if (varianceMetadata) {
        varianceMetadata->combine(*reader.readVarianceMetadata());
    }
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::MaskedImage(ImagePtr image, MaskPtr mask,
                                                                  VariancePtr variance)
        : _image(image != nullptr ? image : std::make_shared<Image>()),
          _mask(mask), _variance(variance) {
    conformSizes();
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::MaskedImage(MaskedImage const& rhs, bool deep)
        : _image(rhs._image), _mask(rhs._mask), _variance(rhs._variance) {
    if (deep) {
        _image = std::shared_ptr<Image>(new Image(*rhs.getImage(), deep));
        _mask = std::shared_ptr<Mask>(new Mask(*rhs.getMask(), deep));
        _variance = std::shared_ptr<Variance>(new Variance(*rhs.getVariance(), deep));
    }
    conformSizes();
}

// Delegate to copy-constructor for backwards compatibility
template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::MaskedImage(MaskedImage&& rhs)
        : MaskedImage(rhs, false) {}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::MaskedImage(MaskedImage const& rhs,
                                                                  const lsst::geom::Box2I& bbox,
                                                                  ImageOrigin const origin, bool deep

                                                                  )
        : _image(new Image(*rhs.getImage(), bbox, origin, deep)),
          _mask(rhs._mask ? new Mask(*rhs.getMask(), bbox, origin, deep) : static_cast<Mask*>(nullptr)),
          _variance(rhs._variance ? new Variance(*rhs.getVariance(), bbox, origin, deep)
                                  : static_cast<Variance*>(nullptr)) {
    conformSizes();
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>& MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::
operator=(MaskedImage const& rhs) = default;

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>& MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::
operator=(MaskedImage&& rhs) = default;

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::swap(MaskedImage& rhs) {
    using std::swap;  // See Meyers, Effective C++, Item 25

    _image.swap(rhs._image);
    _mask.swap(rhs._mask);
    _variance.swap(rhs._variance);
}

// Operators
template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>& MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::
operator=(MaskedImage::Pixel const& rhs) {
    *_image = rhs.image();
    *_mask = rhs.mask();
    *_variance = rhs.variance();

    return *this;
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>& MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::
operator=(MaskedImage::SinglePixel const& rhs) {
    *_image = rhs.image();
    *_mask = rhs.mask();
    *_variance = rhs.variance();

    return *this;
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::assign(MaskedImage const& rhs,
                                                                  lsst::geom::Box2I const& bbox,
                                                                  ImageOrigin origin) {
    _image->assign(*rhs.getImage(), bbox, origin);
    _mask->assign(*rhs.getMask(), bbox, origin);
    _variance->assign(*rhs.getVariance(), bbox, origin);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>& MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::
operator+=(MaskedImage const& rhs) {
    *_image += *rhs.getImage();
    *_mask |= *rhs.getMask();
    *_variance += *rhs.getVariance();
    return *this;
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::scaledPlus(double const c,
                                                                      MaskedImage const& rhs) {
    (*_image).scaledPlus(c, *rhs.getImage());
    *_mask |= *rhs.getMask();
    (*_variance).scaledPlus(c * c, *rhs.getVariance());
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>& MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::
operator+=(ImagePixelT const rhs) {
    *_image += rhs;
    return *this;
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>& MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::
operator-=(MaskedImage const& rhs) {
    *_image -= *rhs.getImage();
    *_mask |= *rhs.getMask();
    *_variance += *rhs.getVariance();
    return *this;
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::scaledMinus(double const c,
                                                                       MaskedImage const& rhs) {
    (*_image).scaledMinus(c, *rhs.getImage());
    *_mask |= *rhs.getMask();
    (*_variance).scaledPlus(c * c, *rhs.getVariance());
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>& MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::
operator-=(ImagePixelT const rhs) {
    *_image -= rhs;
    return *this;
}

namespace {
/// @internal Functor to calculate the variance of the product of two independent variables
template <typename ImagePixelT, typename VariancePixelT>
struct productVariance {
    double operator()(ImagePixelT lhs, ImagePixelT rhs, VariancePixelT varLhs, VariancePixelT varRhs) {
        return lhs * lhs * varRhs + rhs * rhs * varLhs;
    }
};

/// @internal Functor to calculate variance of the product of two independent variables, with the rhs scaled
/// by c
template <typename ImagePixelT, typename VariancePixelT>
struct scaledProductVariance {
    double _c;
    scaledProductVariance(double const c) : _c(c) {}
    double operator()(ImagePixelT lhs, ImagePixelT rhs, VariancePixelT varLhs, VariancePixelT varRhs) {
        return _c * _c * (lhs * lhs * varRhs + rhs * rhs * varLhs);
    }
};
}  // namespace

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>& MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::
operator*=(MaskedImage const& rhs) {
    // Must do variance before we modify the image values
    if (_image->getDimensions() != rhs._image->getDimensions()) {
        throw LSST_EXCEPT(pexExcept::LengthError,
                          boost::str(boost::format("Images are of different size, %dx%d v %dx%d") %
                                     _image->getWidth() % _image->getHeight() % rhs._image->getWidth() % rhs._image->getHeight()));
    }
    transform_pixels(_image->_getRawView(),         // lhs
                     rhs._image->_getRawView(),     // rhs,
                     _variance->_getRawView(),      // Var(lhs),
                     rhs._variance->_getRawView(),  // Var(rhs)
                     _variance->_getRawView(),      // result
                     productVariance<ImagePixelT, VariancePixelT>());

    *_image *= *rhs.getImage();
    *_mask |= *rhs.getMask();
    return *this;
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::scaledMultiplies(double const c,
                                                                            MaskedImage const& rhs) {
    // Must do variance before we modify the image values
    if (_image->getDimensions() != rhs._image->getDimensions()) {
        throw LSST_EXCEPT(pexExcept::LengthError,
                          boost::str(boost::format("Images are of different size, %dx%d v %dx%d") %
                                     _image->getWidth() % _image->getHeight() % rhs._image->getWidth() % rhs._image->getHeight()));
    }
    transform_pixels(_image->_getRawView(),         // lhs
                     rhs._image->_getRawView(),     // rhs,
                     _variance->_getRawView(),      // Var(lhs),
                     rhs._variance->_getRawView(),  // Var(rhs)
                     _variance->_getRawView(),      // result
                     scaledProductVariance<ImagePixelT, VariancePixelT>(c));

    (*_image).scaledMultiplies(c, *rhs.getImage());
    *_mask |= *rhs.getMask();
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>& MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::
operator*=(ImagePixelT const rhs) {
    *_image *= rhs;
    *_variance *= rhs * rhs;
    return *this;
}

namespace {
/// @internal Functor to calculate the variance of the ratio of two independent variables
template <typename ImagePixelT, typename VariancePixelT>
struct quotientVariance {
    double operator()(ImagePixelT lhs, ImagePixelT rhs, VariancePixelT varLhs, VariancePixelT varRhs) {
        ImagePixelT const rhs2 = rhs * rhs;
        return (lhs * lhs * varRhs + rhs2 * varLhs) / (rhs2 * rhs2);
    }
};
/// @internal Functor to calculate the variance of the ratio of two independent variables, the second scaled
/// by c
template <typename ImagePixelT, typename VariancePixelT>
struct scaledQuotientVariance {
    double _c;
    scaledQuotientVariance(double c) : _c(c) {}
    double operator()(ImagePixelT lhs, ImagePixelT rhs, VariancePixelT varLhs, VariancePixelT varRhs) {
        ImagePixelT const rhs2 = rhs * rhs;
        return (lhs * lhs * varRhs + rhs2 * varLhs) / (_c * _c * rhs2 * rhs2);
    }
};
}  // namespace

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>& MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::
operator/=(MaskedImage const& rhs) {
    // Must do variance before we modify the image values
    if (_image->getDimensions() != rhs._image->getDimensions()) {
        throw LSST_EXCEPT(pexExcept::LengthError,
                          boost::str(boost::format("Images are of different size, %dx%d v %dx%d") %
                                     _image->getWidth() % _image->getHeight() % rhs._image->getWidth() % rhs._image->getHeight()));
    }
    transform_pixels(_image->_getRawView(),         // lhs
                     rhs._image->_getRawView(),     // rhs,
                     _variance->_getRawView(),      // Var(lhs),
                     rhs._variance->_getRawView(),  // Var(rhs)
                     _variance->_getRawView(),      // result
                     quotientVariance<ImagePixelT, VariancePixelT>());

    *_image /= *rhs.getImage();
    *_mask |= *rhs.getMask();
    return *this;
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::scaledDivides(double const c,
                                                                         MaskedImage const& rhs) {
    // Must do variance before we modify the image values
    if (_image->getDimensions() != rhs._image->getDimensions()) {
        throw LSST_EXCEPT(pexExcept::LengthError,
                          str(boost::format("Images are of different size, %dx%d v %dx%d") %
                              _image->getWidth() % _image->getHeight() % rhs._image->getWidth() % rhs._image->getHeight()));
    }
    transform_pixels(_image->_getRawView(),         // lhs
                     rhs._image->_getRawView(),     // rhs,
                     _variance->_getRawView(),      // Var(lhs),
                     rhs._variance->_getRawView(),  // Var(rhs)
                     _variance->_getRawView(),      // result
                     scaledQuotientVariance<ImagePixelT, VariancePixelT>(c));

    (*_image).scaledDivides(c, *rhs.getImage());
    *_mask |= *rhs._mask;
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>& MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::
operator/=(ImagePixelT const rhs) {
    *_image /= rhs;
    *_variance /= rhs * rhs;
    return *this;
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::writeFits(
        std::string const& fileName, std::shared_ptr<daf::base::PropertySet const> metadata,
        std::shared_ptr<daf::base::PropertySet const> imageMetadata,
        std::shared_ptr<daf::base::PropertySet const> maskMetadata,
        std::shared_ptr<daf::base::PropertySet const> varianceMetadata) const {
    fits::Fits fitsfile(fileName, "w", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, metadata, imageMetadata, maskMetadata, varianceMetadata);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::writeFits(
        fits::MemFileManager& manager, std::shared_ptr<daf::base::PropertySet const> metadata,
        std::shared_ptr<daf::base::PropertySet const> imageMetadata,
        std::shared_ptr<daf::base::PropertySet const> maskMetadata,
        std::shared_ptr<daf::base::PropertySet const> varianceMetadata) const {
    fits::Fits fitsfile(manager, "w", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, metadata, imageMetadata, maskMetadata, varianceMetadata);
}

namespace {

void processPlaneMetadata(daf::base::PropertySet const * metadata,
                          std::shared_ptr<daf::base::PropertySet>& hdr, char const* exttype) {
    if (metadata) {
        hdr = metadata->deepCopy();
    } else {
        hdr.reset(new daf::base::PropertyList());
    }
    hdr->set("INHERIT", true);
    hdr->set("EXTTYPE", exttype);
    hdr->set("EXTNAME", exttype);
}

}  // namespace

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::writeFits(
        fits::Fits& fitsfile, std::shared_ptr<daf::base::PropertySet const> metadata,
        std::shared_ptr<daf::base::PropertySet const> imageMetadata,
        std::shared_ptr<daf::base::PropertySet const> maskMetadata,
        std::shared_ptr<daf::base::PropertySet const> varianceMetadata) const {
    writeFits(fitsfile, fits::ImageWriteOptions(*_image), fits::ImageWriteOptions(*_mask),
              fits::ImageWriteOptions(*_variance), metadata, imageMetadata, maskMetadata, varianceMetadata);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::writeFits(
        std::string const& fileName, fits::ImageWriteOptions const& imageOptions,
        fits::ImageWriteOptions const& maskOptions, fits::ImageWriteOptions const& varianceOptions,
        std::shared_ptr<daf::base::PropertySet const> metadata,
        std::shared_ptr<daf::base::PropertySet const> imageMetadata,
        std::shared_ptr<daf::base::PropertySet const> maskMetadata,
        std::shared_ptr<daf::base::PropertySet const> varianceMetadata) const {
    fits::Fits fitsfile(fileName, "w", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, imageOptions, maskOptions, varianceOptions, metadata, imageMetadata, maskMetadata,
              varianceMetadata);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::writeFits(
        fits::MemFileManager& manager, fits::ImageWriteOptions const& imageOptions,
        fits::ImageWriteOptions const& maskOptions, fits::ImageWriteOptions const& varianceOptions,
        std::shared_ptr<daf::base::PropertySet const> metadata,
        std::shared_ptr<daf::base::PropertySet const> imageMetadata,
        std::shared_ptr<daf::base::PropertySet const> maskMetadata,
        std::shared_ptr<daf::base::PropertySet const> varianceMetadata) const {
    fits::Fits fitsfile(manager, "w", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, imageOptions, maskOptions, varianceOptions, metadata, imageMetadata, maskMetadata,
              varianceMetadata);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::writeFits(
        fits::Fits& fitsfile, fits::ImageWriteOptions const& imageOptions,
        fits::ImageWriteOptions const& maskOptions, fits::ImageWriteOptions const& varianceOptions,
        std::shared_ptr<daf::base::PropertySet const> metadata,
        std::shared_ptr<daf::base::PropertySet const> imageMetadata,
        std::shared_ptr<daf::base::PropertySet const> maskMetadata,
        std::shared_ptr<daf::base::PropertySet const> varianceMetadata) const {
    std::shared_ptr<daf::base::PropertySet> header;
    if (metadata) {
        header = metadata->deepCopy();
    } else {
        header = std::make_shared<daf::base::PropertyList>();
    }

    if (fitsfile.countHdus() != 0) {
        throw LSST_EXCEPT(pex::exceptions::LogicError,
                          "MaskedImage::writeFits can only write to an empty file");
    }
    if (fitsfile.getHdu() < 1) {
        // Don't ever write images to primary; instead we make an empty primary.
        fitsfile.createEmpty();
    } else {
        fitsfile.setHdu(0);
    }
    // Primary HDU should not include an EXTNAME header.
    // On read we merge HDUs and this can lead to an EXTNAME being propagated.
    if (header->exists("EXTNAME")) {
        header->remove("EXTNAME");
    }
    fitsfile.writeMetadata(*header);

    processPlaneMetadata(imageMetadata.get(), header, "IMAGE");
    _image->writeFits(fitsfile, imageOptions, header.get(), _mask.get());

    processPlaneMetadata(maskMetadata.get(), header, "MASK");
    _mask->writeFits(fitsfile, maskOptions, header.get());

    processPlaneMetadata(varianceMetadata.get(), header, "VARIANCE");
    _variance->writeFits(fitsfile, varianceOptions, header.get(), _mask.get());
}

// private function conformSizes() ensures that the Mask and Variance have the same dimensions
// as Image.  If Mask and/or Variance have non-zero dimensions that conflict with the size of Image,
// a lsst::pex::exceptions::LengthError is thrown.

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::conformSizes() {
    if (!_mask || _mask->getWidth() == 0 || _mask->getHeight() == 0) {
        _mask = MaskPtr(new Mask(_image->getBBox()));
        *_mask = 0;
    } else {
        if (_mask->getDimensions() != _image->getDimensions()) {
            throw LSST_EXCEPT(
                    pex::exceptions::LengthError,
                    (boost::format("Dimension mismatch: Image %dx%d v. Mask %dx%d") % _image->getWidth() %
                     _image->getHeight() % _mask->getWidth() % _mask->getHeight())
                            .str());
        }
    }

    if (!_variance || _variance->getWidth() == 0 || _variance->getHeight() == 0) {
        _variance = VariancePtr(new Variance(_image->getBBox()));
        *_variance = 0;
    } else {
        if (_variance->getDimensions() != _image->getDimensions()) {
            throw LSST_EXCEPT(
                    pex::exceptions::LengthError,
                    (boost::format("Dimension mismatch: Image %dx%d v. Variance %dx%d") % _image->getWidth() %
                     _image->getHeight() % _variance->getWidth() % _variance->getHeight())
                            .str());
        }
    }
}

//
// Iterators and locators
//
template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
typename MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::iterator
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::begin() const {
#if 0  // this doesn't compile; why?
    return iterator(_image->begin(), _mask->begin(), _variance->begin());
#else
    typename Image::iterator imageBegin = _image->begin();
    typename Mask::iterator maskBegin = _mask->begin();
    typename Variance::iterator varianceBegin = _variance->begin();

    return iterator(imageBegin, maskBegin, varianceBegin);
#endif
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
typename MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::iterator
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::end() const {
    typename Image::iterator imageEnd = getImage()->end();
    typename Mask::iterator maskEnd = getMask()->end();
    typename Variance::iterator varianceEnd = getVariance()->end();

    return iterator(imageEnd, maskEnd, varianceEnd);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
typename MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::iterator
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::at(int const x, int const y) const {
    typename Image::iterator imageEnd = getImage()->at(x, y);
    typename Mask::iterator maskEnd = getMask()->at(x, y);
    typename Variance::iterator varianceEnd = getVariance()->at(x, y);

    return iterator(imageEnd, maskEnd, varianceEnd);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
typename MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::reverse_iterator
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::rbegin() const {
    typename Image::reverse_iterator imageBegin = _image->rbegin();
    typename Mask::reverse_iterator maskBegin = _mask->rbegin();
    typename Variance::reverse_iterator varianceBegin = _variance->rbegin();

    return reverse_iterator(imageBegin, maskBegin, varianceBegin);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
typename MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::reverse_iterator
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::rend() const {
    typename Image::reverse_iterator imageEnd = getImage()->rend();
    typename Mask::reverse_iterator maskEnd = getMask()->rend();
    typename Variance::reverse_iterator varianceEnd = getVariance()->rend();

    return reverse_iterator(imageEnd, maskEnd, varianceEnd);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
typename MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::x_iterator
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::row_begin(int y) const {
    typename Image::x_iterator imageBegin = _image->row_begin(y);
    typename Mask::x_iterator maskBegin = _mask->row_begin(y);
    typename Variance::x_iterator varianceBegin = _variance->row_begin(y);

    return x_iterator(imageBegin, maskBegin, varianceBegin);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
typename MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::x_iterator
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::row_end(int y) const {
    typename Image::x_iterator imageEnd = getImage()->row_end(y);
    typename Mask::x_iterator maskEnd = getMask()->row_end(y);
    typename Variance::x_iterator varianceEnd = getVariance()->row_end(y);

    return x_iterator(imageEnd, maskEnd, varianceEnd);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
typename MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::y_iterator
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::col_begin(int x) const {
    typename Image::y_iterator imageBegin = _image->col_begin(x);
    typename Mask::y_iterator maskBegin = _mask->col_begin(x);
    typename Variance::y_iterator varianceBegin = _variance->col_begin(x);

    return y_iterator(imageBegin, maskBegin, varianceBegin);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
typename MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::y_iterator
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::col_end(int x) const {
    typename Image::y_iterator imageEnd = getImage()->col_end(x);
    typename Mask::y_iterator maskEnd = getMask()->col_end(x);
    typename Variance::y_iterator varianceEnd = getVariance()->col_end(x);

    return y_iterator(imageEnd, maskEnd, varianceEnd);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
typename MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::fast_iterator
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::begin(bool contiguous) const {
    typename Image::fast_iterator imageBegin = _image->begin(contiguous);
    typename Mask::fast_iterator maskBegin = _mask->begin(contiguous);
    typename Variance::fast_iterator varianceBegin = _variance->begin(contiguous);

    return fast_iterator(imageBegin, maskBegin, varianceBegin);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
typename MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::fast_iterator
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::end(bool contiguous) const {
    typename Image::fast_iterator imageEnd = getImage()->end(contiguous);
    typename Mask::fast_iterator maskEnd = getMask()->end(contiguous);
    typename Variance::fast_iterator varianceEnd = getVariance()->end(contiguous);

    return fast_iterator(imageEnd, maskEnd, varianceEnd);
}

template <typename ImagePixelT1, typename ImagePixelT2>
bool imagesOverlap(MaskedImage<ImagePixelT1, MaskPixel, VariancePixel> const& image1,
                   MaskedImage<ImagePixelT2, MaskPixel, VariancePixel> const& image2) {
    return imagesOverlap(*image1.getImage(), *image2.getImage()) ||
           imagesOverlap(*image1.getVariance(), *image2.getVariance()) ||
           imagesOverlap(*image1.getMask(), *image2.getMask());
}

//
// Explicit instantiations
//
#define INSTANTIATE2(ImagePixelT1, ImagePixelT2)                                              \
    template bool imagesOverlap<ImagePixelT1, ImagePixelT2>(MaskedImage<ImagePixelT1> const&, \
                                                            MaskedImage<ImagePixelT2> const&);

template class MaskedImage<std::uint16_t>;
template class MaskedImage<int>;
template class MaskedImage<float>;
template class MaskedImage<double>;
template class MaskedImage<std::uint64_t>;

INSTANTIATE2(std::uint16_t, std::uint16_t);
INSTANTIATE2(std::uint16_t, int);
INSTANTIATE2(std::uint16_t, float);
INSTANTIATE2(std::uint16_t, double);
INSTANTIATE2(std::uint16_t, std::uint64_t);

INSTANTIATE2(int, std::uint16_t);
INSTANTIATE2(int, int);
INSTANTIATE2(int, float);
INSTANTIATE2(int, double);
INSTANTIATE2(int, std::uint64_t);

INSTANTIATE2(float, std::uint16_t);
INSTANTIATE2(float, int);
INSTANTIATE2(float, float);
INSTANTIATE2(float, double);
INSTANTIATE2(float, std::uint64_t);

INSTANTIATE2(double, std::uint16_t);
INSTANTIATE2(double, int);
INSTANTIATE2(double, float);
INSTANTIATE2(double, double);
INSTANTIATE2(double, std::uint64_t);

INSTANTIATE2(std::uint64_t, std::uint16_t);
INSTANTIATE2(std::uint64_t, int);
INSTANTIATE2(std::uint64_t, float);
INSTANTIATE2(std::uint64_t, double);
INSTANTIATE2(std::uint64_t, std::uint64_t);

}  // namespace image
}  // namespace afw
}  // namespace lsst
