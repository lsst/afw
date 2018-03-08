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
#include <typeinfo>
#include <sys/stat.h>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic pop
#include "boost/regex.hpp"
#include "boost/filesystem/path.hpp"
#include "lsst/log/Log.h"
#include "lsst/pex/exceptions.h"
#include "boost/algorithm/string/trim.hpp"

#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/image/fits/fits_io.h"
#include "lsst/afw/fits.h"

namespace lsst {
namespace afw {
namespace image {

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::MaskedImage(unsigned int width, unsigned int height,
                                                                  MaskPlaneDict const& planeDict)
        : daf::base::Citizen(typeid(this)),
          _image(new Image(width, height)),
          _mask(new Mask(width, height, planeDict)),
          _variance(new Variance(width, height)) {
    *_image = 0;
    *_mask = 0x0;
    *_variance = 0;
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::MaskedImage(geom::Extent2I const& dimensions,
                                                                  MaskPlaneDict const& planeDict)
        : daf::base::Citizen(typeid(this)),
          _image(new Image(dimensions)),
          _mask(new Mask(dimensions, planeDict)),
          _variance(new Variance(dimensions)) {
    *_image = 0;
    *_mask = 0x0;
    *_variance = 0;
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::MaskedImage(geom::Box2I const& bbox,
                                                                  MaskPlaneDict const& planeDict)
        : daf::base::Citizen(typeid(this)),
          _image(new Image(bbox)),
          _mask(new Mask(bbox, planeDict)),
          _variance(new Variance(bbox)) {
    *_image = 0;
    *_mask = 0x0;
    *_variance = 0;
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::MaskedImage(
        std::string const& fileName, std::shared_ptr<daf::base::PropertySet> metadata,
        geom::Box2I const& bbox, ImageOrigin origin, bool conformMasks, bool needAllHdus,
        std::shared_ptr<daf::base::PropertySet> imageMetadata,
        std::shared_ptr<daf::base::PropertySet> maskMetadata,
        std::shared_ptr<daf::base::PropertySet> varianceMetadata)
        : daf::base::Citizen(typeid(this)), _image(), _mask(), _variance() {
    fits::Fits fitsfile(fileName, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    *this = MaskedImage(fitsfile, metadata, bbox, origin, conformMasks, needAllHdus, imageMetadata,
                        maskMetadata, varianceMetadata);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::MaskedImage(
        fits::MemFileManager& manager, std::shared_ptr<daf::base::PropertySet> metadata,
        geom::Box2I const& bbox, ImageOrigin origin, bool conformMasks, bool needAllHdus,
        std::shared_ptr<daf::base::PropertySet> imageMetadata,
        std::shared_ptr<daf::base::PropertySet> maskMetadata,
        std::shared_ptr<daf::base::PropertySet> varianceMetadata)
        : daf::base::Citizen(typeid(this)), _image(), _mask(), _variance() {
    fits::Fits fitsfile(manager, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    *this = MaskedImage(fitsfile, metadata, bbox, origin, conformMasks, needAllHdus, imageMetadata,
                        maskMetadata, varianceMetadata);
}

namespace {

// Helper functions for MaskedImage FITS ctor.

void checkExtType(fits::Fits& fitsfile, std::shared_ptr<daf::base::PropertySet> metadata,
                  std::string const& expected) {
    try {
        std::string exttype = boost::algorithm::trim_right_copy(metadata->getAsString("EXTTYPE"));
        if (exttype != "" && exttype != expected) {
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                              (boost::format("Reading %s (hdu %d) Expected EXTTYPE==\"%s\", saw \"%s\"") %
                               expected % fitsfile.getFileName() % fitsfile.getHdu() % exttype)
                                      .str());
        }
        metadata->remove("EXTTYPE");
    } catch (pex::exceptions::NotFoundError) {
        LOGLS_WARN("afw.image.MaskedImage", "Expected extension type not found: " << expected);
    }
}

void ensureMetadata(std::shared_ptr<daf::base::PropertySet>& metadata) {
    if (!metadata) {
        metadata.reset(new daf::base::PropertyList());
    }
}

}  // anonymous

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::MaskedImage(
        fits::Fits& fitsfile, std::shared_ptr<daf::base::PropertySet> metadata, geom::Box2I const& bbox,
        ImageOrigin origin, bool conformMasks, bool needAllHdus,
        std::shared_ptr<daf::base::PropertySet> imageMetadata,
        std::shared_ptr<daf::base::PropertySet> maskMetadata,
        std::shared_ptr<daf::base::PropertySet> varianceMetadata)
        : daf::base::Citizen(typeid(this)), _image(), _mask(), _variance() {
    // When reading a standard Masked Image, we expect four HDUs:
    // * The primary (HDU 0) is empty;
    // * The first extension (HDU 1) contains the image data;
    // * The second extension (HDU 2) contains mask data;
    // * The third extension (HDU 3) contains the variance.
    //
    // If the image HDU is unreadable, we will throw.
    //
    // If the user has specified a non-default HDU, we load image data from
    // that HDU, but do not attempt to load mask/variance data; rather, log a
    // warning and return (blank) defaults.
    //
    // If the mask and/or variance is unreadable, we log a warning and return
    // (blank) defaults.

    LOG_LOGGER _log = LOG_GET("afw.image.MaskedImage");

    enum class Hdu { Primary = 0, Image, Mask, Variance };

    // If the user has requested a non-default HDU and we require all HDUs, we fail.
    if (needAllHdus && fitsfile.getHdu() > static_cast<int>(Hdu::Image)) {
        throw LSST_EXCEPT(fits::FitsError, "Cannot read all HDUs starting from non-default");
    }

    if (metadata) {
        // Read primary metadata - only if user asks for it.
        // If the primary HDU is not empty, this may be the same as imageMetadata.
        auto prevHdu = fitsfile.getHdu();
        fitsfile.setHdu(static_cast<int>(Hdu::Primary));
        fitsfile.readMetadata(*metadata);
        fitsfile.setHdu(prevHdu);
    }

    // setHdu(fits::DEFAULT_HDU) jumps to the first extension iff the primary HDU is both
    // empty and currently selected.
    fitsfile.setHdu(fits::DEFAULT_HDU);
    ensureMetadata(imageMetadata);
    _image.reset(new Image(fitsfile, imageMetadata, bbox, origin));
    checkExtType(fitsfile, imageMetadata, "IMAGE");

    if (fitsfile.getHdu() != static_cast<int>(Hdu::Image)) {
        // Reading the image from a non-default HDU means we do not attempt to
        // read mask and variance.
        _mask.reset(new Mask(_image->getBBox()));
        _variance.reset(new Variance(_image->getBBox()));
    } else {
        try {
            fitsfile.setHdu(static_cast<int>(Hdu::Mask));
            ensureMetadata(maskMetadata);
            _mask.reset(new Mask(fitsfile, maskMetadata, bbox, origin, conformMasks));
            checkExtType(fitsfile, maskMetadata, "MASK");
        } catch (fits::FitsError& e) {
            if (needAllHdus) {
                LSST_EXCEPT_ADD(e, "Reading Mask");
                throw e;
            }
            LOGLS_WARN(_log, "Mask unreadable (" << e << "); using default");
            // By resetting the status we are able to read the next HDU (the variance).
            fitsfile.status = 0;
            _mask.reset(new Mask(_image->getBBox()));
        }

        try {
            fitsfile.setHdu(static_cast<int>(Hdu::Variance));
            ensureMetadata(varianceMetadata);
            _variance.reset(new Variance(fitsfile, varianceMetadata, bbox, origin));
            checkExtType(fitsfile, varianceMetadata, "VARIANCE");
        } catch (fits::FitsError& e) {
            if (needAllHdus) {
                LSST_EXCEPT_ADD(e, "Reading Variance");
                throw e;
            }
            LOGLS_WARN(_log, "Variance unreadable (" << e << "); using default");
            fitsfile.status = 0;
            _variance.reset(new Variance(_image->getBBox()));
        }
    }
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::MaskedImage(ImagePtr image, MaskPtr mask,
                                                                  VariancePtr variance)
        : daf::base::Citizen(typeid(this)), _image(image), _mask(mask), _variance(variance) {
    conformSizes();
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::MaskedImage(MaskedImage const& rhs, bool deep)
        : daf::base::Citizen(typeid(this)), _image(rhs._image), _mask(rhs._mask), _variance(rhs._variance) {
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
                                                                  const geom::Box2I& bbox,
                                                                  ImageOrigin const origin, bool deep

                                                                  )
        : daf::base::Citizen(typeid(this)),
          _image(new Image(*rhs.getImage(), bbox, origin, deep)),
          _mask(rhs._mask ? new Mask(*rhs.getMask(), bbox, origin, deep) : static_cast<Mask*>(NULL)),
          _variance(rhs._variance ? new Variance(*rhs.getVariance(), bbox, origin, deep)
                                  : static_cast<Variance*>(NULL)) {
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
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>& MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::
operator<<=(MaskedImage const& rhs) {
    assign(rhs);
    return *this;
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::assign(MaskedImage const& rhs,
                                                                  geom::Box2I const& bbox,
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
    explicit scaledProductVariance(double const c) : _c(c) {}
    double operator()(ImagePixelT lhs, ImagePixelT rhs, VariancePixelT varLhs, VariancePixelT varRhs) {
        return _c * _c * (lhs * lhs * varRhs + rhs * rhs * varLhs);
    }
};
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>& MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::
operator*=(MaskedImage const& rhs) {
    // Must do variance before we modify the image values
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
    explicit scaledQuotientVariance(double c) : _c(c) {}
    double operator()(ImagePixelT lhs, ImagePixelT rhs, VariancePixelT varLhs, VariancePixelT varRhs) {
        ImagePixelT const rhs2 = rhs * rhs;
        return (lhs * lhs * varRhs + rhs2 * varLhs) / (_c * _c * rhs2 * rhs2);
    }
};
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>& MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::
operator/=(MaskedImage const& rhs) {
    // Must do variance before we modify the image values
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

void processPlaneMetadata(std::shared_ptr<daf::base::PropertySet const> metadata,
                          std::shared_ptr<daf::base::PropertySet>& hdr, char const* exttype) {
    if (metadata) {
        hdr = metadata->deepCopy();
    } else {
        hdr.reset(new daf::base::PropertyList());
    }
    hdr->set("INHERIT", true);
    hdr->set("EXTTYPE", exttype);
}

}  // anonymous

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
    std::string const& fileName,
    fits::ImageWriteOptions const& imageOptions,
    fits::ImageWriteOptions const& maskOptions,
    fits::ImageWriteOptions const& varianceOptions,
    std::shared_ptr<daf::base::PropertySet const> metadata,
    std::shared_ptr<daf::base::PropertySet const> imageMetadata,
    std::shared_ptr<daf::base::PropertySet const> maskMetadata,
    std::shared_ptr<daf::base::PropertySet const> varianceMetadata
) const {
    fits::Fits fitsfile(fileName, "w", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, imageOptions, maskOptions, varianceOptions, metadata, imageMetadata,
              maskMetadata, varianceMetadata);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::writeFits(
    fits::MemFileManager& manager,
    fits::ImageWriteOptions const& imageOptions,
    fits::ImageWriteOptions const& maskOptions,
    fits::ImageWriteOptions const& varianceOptions,
    std::shared_ptr<daf::base::PropertySet const> metadata,
    std::shared_ptr<daf::base::PropertySet const> imageMetadata,
    std::shared_ptr<daf::base::PropertySet const> maskMetadata,
    std::shared_ptr<daf::base::PropertySet const> varianceMetadata
) const {
    fits::Fits fitsfile(manager, "w", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, imageOptions, maskOptions, varianceOptions, metadata, imageMetadata,
              maskMetadata, varianceMetadata);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>::writeFits(
    fits::Fits& fitsfile,
    fits::ImageWriteOptions const& imageOptions,
    fits::ImageWriteOptions const& maskOptions,
    fits::ImageWriteOptions const& varianceOptions,
    std::shared_ptr<daf::base::PropertySet const> metadata,
    std::shared_ptr<daf::base::PropertySet const> imageMetadata,
    std::shared_ptr<daf::base::PropertySet const> maskMetadata,
    std::shared_ptr<daf::base::PropertySet const> varianceMetadata
) const {
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
    fitsfile.writeMetadata(*header);

    processPlaneMetadata(imageMetadata, header, "IMAGE");
    _image->writeFits(fitsfile, imageOptions, header, _mask);

    processPlaneMetadata(maskMetadata, header, "MASK");
    _mask->writeFits(fitsfile, maskOptions, header);

    processPlaneMetadata(varianceMetadata, header, "VARIANCE");
    _variance->writeFits(fitsfile, varianceOptions, header, _mask);
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

//
// Explicit instantiations
//
template class MaskedImage<std::uint16_t>;
template class MaskedImage<int>;
template class MaskedImage<float>;
template class MaskedImage<double>;
template class MaskedImage<std::uint64_t>;
}
}
}  // end lsst::afw::image
