// -*- LSST-C++ -*- // fixed format comment for emacs

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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

#include <memory>
#include <stdexcept>
#include <sstream>
#include <cstdint>

#include "boost/format.hpp"
#include "boost/algorithm/string/trim.hpp"

#include "lsst/daf/base/PropertySet.h"
#include "lsst/daf/base/PropertyList.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/geom/SkyWcs.h"
#include "lsst/afw/image/Exposure.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/image/Calib.h"
#include "lsst/afw/math/ConvolveImage.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/fits.h"
#include "lsst/afw/image/ExposureFitsReader.h"

namespace lsst {
namespace afw {
namespace image {

// CLASS CONSTRUCTORS and DESTRUCTOR

template <typename ImageT, typename MaskT, typename VarianceT>
Exposure<ImageT, MaskT, VarianceT>::Exposure(unsigned int width, unsigned int height,
                                             std::shared_ptr<geom::SkyWcs const> wcs)
        : daf::base::Citizen(typeid(this)), _maskedImage(width, height), _info(new ExposureInfo(wcs)) {}

template <typename ImageT, typename MaskT, typename VarianceT>
Exposure<ImageT, MaskT, VarianceT>::Exposure(lsst::geom::Extent2I const &dimensions,
                                             std::shared_ptr<geom::SkyWcs const> wcs)
        : daf::base::Citizen(typeid(this)), _maskedImage(dimensions), _info(new ExposureInfo(wcs)) {}

template <typename ImageT, typename MaskT, typename VarianceT>
Exposure<ImageT, MaskT, VarianceT>::Exposure(lsst::geom::Box2I const &bbox,
                                             std::shared_ptr<geom::SkyWcs const> wcs)
        : daf::base::Citizen(typeid(this)), _maskedImage(bbox), _info(new ExposureInfo(wcs)) {}

template <typename ImageT, typename MaskT, typename VarianceT>
Exposure<ImageT, MaskT, VarianceT>::Exposure(MaskedImageT &maskedImage,
                                             std::shared_ptr<geom::SkyWcs const> wcs)
        : daf::base::Citizen(typeid(this)), _maskedImage(maskedImage), _info(new ExposureInfo(wcs)) {}

template <typename ImageT, typename MaskT, typename VarianceT>
Exposure<ImageT, MaskT, VarianceT>::Exposure(MaskedImageT &maskedImage, std::shared_ptr<ExposureInfo> info)
        : daf::base::Citizen(typeid(this)),
          _maskedImage(maskedImage),
          _info(info ? info : std::make_shared<ExposureInfo>()) {}

template <typename ImageT, typename MaskT, typename VarianceT>
Exposure<ImageT, MaskT, VarianceT>::Exposure(Exposure const &src, bool const deep)
        : daf::base::Citizen(typeid(this)),
          _maskedImage(src.getMaskedImage(), deep),
          _info(new ExposureInfo(*src.getInfo(), deep)) {}
// Delegate to copy-constructor for backwards compatibility
template <typename ImageT, typename MaskT, typename VarianceT>
Exposure<ImageT, MaskT, VarianceT>::Exposure(Exposure &&src) : Exposure(src) {}

template <typename ImageT, typename MaskT, typename VarianceT>
Exposure<ImageT, MaskT, VarianceT>::Exposure(Exposure const &src, lsst::geom::Box2I const &bbox,
                                             ImageOrigin const origin, bool const deep)
        : daf::base::Citizen(typeid(this)),
          _maskedImage(src.getMaskedImage(), bbox, origin, deep),
          _info(new ExposureInfo(*src.getInfo(), deep)) {}

template <typename ImageT, typename MaskT, typename VarianceT>
Exposure<ImageT, MaskT, VarianceT>::Exposure(std::string const &fileName, lsst::geom::Box2I const &bbox,
                                             ImageOrigin origin, bool conformMasks)
        : daf::base::Citizen(typeid(this)), _maskedImage(), _info(new ExposureInfo()) {
    ExposureFitsReader reader(fileName);
    *this = reader.read<ImageT, MaskT, VarianceT>(bbox, origin, conformMasks);
}

template <typename ImageT, typename MaskT, typename VarianceT>
Exposure<ImageT, MaskT, VarianceT>::Exposure(fits::MemFileManager &manager, lsst::geom::Box2I const &bbox,
                                             ImageOrigin origin, bool conformMasks)
        : daf::base::Citizen(typeid(this)), _maskedImage(), _info(new ExposureInfo()) {
    ExposureFitsReader reader(manager);
    *this = reader.read<ImageT, MaskT, VarianceT>(bbox, origin, conformMasks);
}

template <typename ImageT, typename MaskT, typename VarianceT>
Exposure<ImageT, MaskT, VarianceT>::Exposure(fits::Fits &fitsfile, lsst::geom::Box2I const &bbox,
                                             ImageOrigin origin, bool conformMasks)
        : daf::base::Citizen(typeid(this)) {
    ExposureFitsReader reader(&fitsFile);
    *this = reader.read<ImageT, MaskT, VarianceT>(bbox, origin, conformMasks);
}

template <typename ImageT, typename MaskT, typename VarianceT>
Exposure<ImageT, MaskT, VarianceT>::~Exposure() = default;

// SET METHODS

template <typename ImageT, typename MaskT, typename VarianceT>
void Exposure<ImageT, MaskT, VarianceT>::setMaskedImage(MaskedImageT &maskedImage) {
    _maskedImage = maskedImage;
}

template <typename ImageT, typename MaskT, typename VarianceT>
void Exposure<ImageT, MaskT, VarianceT>::setXY0(lsst::geom::Point2I const &origin) {
    lsst::geom::Point2I old(_maskedImage.getXY0());
    if (_info->hasWcs()) {
        auto shift = lsst::geom::Extent2D(origin - old);
        auto newWcs = _info->getWcs()->copyAtShiftedPixelOrigin(shift);
        _info->setWcs(newWcs);
    }
    _maskedImage.setXY0(origin);
}

template <typename ImageT, typename MaskT, typename VarianceT>
Exposure<ImageT, MaskT, VarianceT> &Exposure<ImageT, MaskT, VarianceT>::operator=(Exposure const &) = default;
template <typename ImageT, typename MaskT, typename VarianceT>
Exposure<ImageT, MaskT, VarianceT> &Exposure<ImageT, MaskT, VarianceT>::operator=(Exposure &&) = default;

// Write FITS

template <typename ImageT, typename MaskT, typename VarianceT>
void Exposure<ImageT, MaskT, VarianceT>::writeFits(std::string const &fileName) const {
    fits::Fits fitsfile(fileName, "w", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile);
}

template <typename ImageT, typename MaskT, typename VarianceT>
void Exposure<ImageT, MaskT, VarianceT>::writeFits(fits::MemFileManager &manager) const {
    fits::Fits fitsfile(manager, "w", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile);
}

template <typename ImageT, typename MaskT, typename VarianceT>
void Exposure<ImageT, MaskT, VarianceT>::writeFits(fits::Fits &fitsfile) const {
    writeFits(fitsfile, fits::ImageWriteOptions(*_maskedImage.getImage()),
              fits::ImageWriteOptions(*_maskedImage.getMask()),
              fits::ImageWriteOptions(*_maskedImage.getVariance()));
}

template <typename ImageT, typename MaskT, typename VarianceT>
void Exposure<ImageT, MaskT, VarianceT>::writeFits(std::string const &fileName,
                                                   fits::ImageWriteOptions const &imageOptions,
                                                   fits::ImageWriteOptions const &maskOptions,
                                                   fits::ImageWriteOptions const &varianceOptions) const {
    fits::Fits fitsfile(fileName, "w", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, imageOptions, maskOptions, varianceOptions);
}

template <typename ImageT, typename MaskT, typename VarianceT>
void Exposure<ImageT, MaskT, VarianceT>::writeFits(fits::MemFileManager &manager,
                                                   fits::ImageWriteOptions const &imageOptions,
                                                   fits::ImageWriteOptions const &maskOptions,
                                                   fits::ImageWriteOptions const &varianceOptions) const {
    fits::Fits fitsfile(manager, "w", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, imageOptions, maskOptions, varianceOptions);
}

template <typename ImageT, typename MaskT, typename VarianceT>
void Exposure<ImageT, MaskT, VarianceT>::writeFits(fits::Fits &fitsfile,
                                                   fits::ImageWriteOptions const &imageOptions,
                                                   fits::ImageWriteOptions const &maskOptions,
                                                   fits::ImageWriteOptions const &varianceOptions) const {
    ExposureInfo::FitsWriteData data = _info->_startWriteFits(getXY0());
    _maskedImage.writeFits(fitsfile, imageOptions, maskOptions, varianceOptions, data.metadata,
                           data.imageMetadata, data.maskMetadata, data.varianceMetadata);
    _info->_finishWriteFits(fitsfile, data);
}

namespace {
/**
 * Copy all overlapping pixels from one Exposure to another.
 *
 * If no pixels overlap, ``destination`` shall not be modified.
 *
 * @param destination The Exposure to copy pixels to.
 * @param source The Exposure whose pixels will be copied.
 */
template <class ExposureT>
void _copyCommonPixels(ExposureT &destination, ExposureT const &source) {
    lsst::geom::Box2I overlapBox = destination.getBBox();
    overlapBox.clip(source.getBBox());

    // MaskedImage::assign interprets empty bounding box as "whole image"
    if (!overlapBox.isEmpty()) {
        typename ExposureT::MaskedImageT overlapPixels(source.getMaskedImage(), overlapBox);
        destination.getMaskedImage().assign(overlapPixels, overlapBox);
    }
}
}  // namespace

template <typename ImageT, typename MaskT, typename VarianceT>
Exposure<ImageT, MaskT, VarianceT> Exposure<ImageT, MaskT, VarianceT>::getCutout(
        lsst::geom::SpherePoint const &center, lsst::geom::Extent2I const &size) const {
    if (!hasWcs()) {
        throw LSST_EXCEPT(pex::exceptions::LogicError, "Cannot look up source position without WCS.");
    }
    lsst::geom::Point2D pixelCenter = getWcs()->skyToPixel(center);

    if (!lsst::geom::Box2D(getBBox()).contains(pixelCenter)) {
        std::stringstream buffer;
        buffer << "Point " << center << " lies at pixel " << pixelCenter << ", which lies outside Exposure "
               << getBBox();
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, buffer.str());
    }
    if (size[0] <= 0 || size[1] <= 0) {
        std::stringstream buffer;
        buffer << "Cannot create bounding box with dimensions " << size;
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, buffer.str());
    }
    lsst::geom::Box2I bbox = lsst::geom::Box2I::makeCenteredBox(pixelCenter, size);

    // cutout must have independent ExposureInfo
    auto copyInfo = std::make_shared<ExposureInfo>(*getInfo());
    MaskedImageT blank(bbox);  // Can't initialize Exposure with a temporary
    blank = math::edgePixel<MaskedImageT>(
            typename image::detail::image_traits<MaskedImageT>::image_category());
    Exposure cutout(blank, copyInfo);

    _copyCommonPixels(cutout, *this);
    return cutout;
}

// Explicit instantiations
/// @cond
template class Exposure<std::uint16_t>;
template class Exposure<int>;
template class Exposure<float>;
template class Exposure<double>;
template class Exposure<std::uint64_t>;
/// @endcond
}  // namespace image
}  // namespace afw
}  // namespace lsst
