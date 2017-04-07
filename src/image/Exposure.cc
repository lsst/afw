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

#include <stdexcept>

#include "boost/format.hpp"
#include <memory>
#include <cstdint>
#include "boost/algorithm/string/trim.hpp"

#include "lsst/daf/base/PropertySet.h"
#include "lsst/daf/base/PropertyList.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/image/Exposure.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/image/Calib.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/fits.h"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace afwDetection = lsst::afw::detection;
namespace cameraGeom = lsst::afw::cameraGeom;


// CLASS CONSTRUCTORS and DESTRUCTOR


template<typename ImageT, typename MaskT, typename VarianceT>
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    unsigned int width,
    unsigned int height,
    CONST_PTR(Wcs) wcs
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(width, height),
    _info(new ExposureInfo(wcs))
{}

template<typename ImageT, typename MaskT, typename VarianceT>
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    afwGeom::Extent2I const & dimensions,
    CONST_PTR(Wcs) wcs
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(dimensions),
    _info(new ExposureInfo(wcs))
{}

template<typename ImageT, typename MaskT, typename VarianceT>
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    afwGeom::Box2I const & bbox,
    CONST_PTR(Wcs) wcs
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(bbox),
    _info(new ExposureInfo(wcs))
{}

template<typename ImageT, typename MaskT, typename VarianceT>
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    MaskedImageT &maskedImage,
    CONST_PTR(Wcs) wcs
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(maskedImage),
    _info(new ExposureInfo(wcs))
{}

template<typename ImageT, typename MaskT, typename VarianceT>
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    MaskedImageT & maskedImage,
    PTR(ExposureInfo) info
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(maskedImage),
    _info(info ? info : std::make_shared<ExposureInfo>())
{}

template<typename ImageT, typename MaskT, typename VarianceT>
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    Exposure const &src,
    bool const deep
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(src.getMaskedImage(), deep),
    _info(new ExposureInfo(*src.getInfo(), deep))
{}

template<typename ImageT, typename MaskT, typename VarianceT>
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    Exposure const &src,
    afwGeom::Box2I const& bbox,
    ImageOrigin const origin,
    bool const deep
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(src.getMaskedImage(), bbox, origin, deep),
    _info(new ExposureInfo(*src.getInfo(), deep))
{}

template<typename ImageT, typename MaskT, typename VarianceT>
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    std::string const & fileName, afwGeom::Box2I const& bbox,
    ImageOrigin origin, bool conformMasks
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(),
    _info(new ExposureInfo())
{
    fits::Fits fitsfile(fileName, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    _readFits(fitsfile, bbox, origin, conformMasks);
}

template<typename ImageT, typename MaskT, typename VarianceT>
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    fits::MemFileManager & manager, afwGeom::Box2I const & bbox,
    ImageOrigin origin, bool conformMasks
) :
    lsst::daf::base::Citizen(typeid(this)),
    _maskedImage(),
    _info(new ExposureInfo())
{
    fits::Fits fitsfile(manager, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    _readFits(fitsfile, bbox, origin, conformMasks);
}

template<typename ImageT, typename MaskT, typename VarianceT>
afwImage::Exposure<ImageT, MaskT, VarianceT>::Exposure(
    fits::Fits & fitsfile, afwGeom::Box2I const & bbox,
    ImageOrigin origin, bool conformMasks
) :
    lsst::daf::base::Citizen(typeid(this))
{
    _readFits(fitsfile, bbox, origin, conformMasks);
}

template<typename ImageT, typename MaskT, typename VarianceT>
void afwImage::Exposure<ImageT, MaskT, VarianceT>::_readFits(
    fits::Fits & fitsfile, afwGeom::Box2I const & bbox,
    ImageOrigin origin, bool conformMasks
) {
    PTR(daf::base::PropertySet) metadata(new lsst::daf::base::PropertyList());
    PTR(daf::base::PropertySet) imageMetadata(new lsst::daf::base::PropertyList());
    _maskedImage = MaskedImageT(fitsfile, metadata, bbox, origin, conformMasks, false, imageMetadata);
    _info->_readFits(fitsfile, metadata, imageMetadata);
}


template<typename ImageT, typename MaskT, typename VarianceT>
afwImage::Exposure<ImageT, MaskT, VarianceT>::~Exposure(){}

// SET METHODS

template<typename ImageT, typename MaskT, typename VarianceT>
void afwImage::Exposure<ImageT, MaskT, VarianceT>::setMaskedImage(MaskedImageT &maskedImage){
    _maskedImage = maskedImage;
}

template<typename ImageT, typename MaskT, typename VarianceT>
void afwImage::Exposure<ImageT, MaskT, VarianceT>::setXY0(afwGeom::Point2I const& origin) {
    afwGeom::Point2I old(_maskedImage.getXY0());
    if (_info->hasWcs())
        _info->getWcs()->shiftReferencePixel(origin.getX() - old.getX(), origin.getY() - old.getY());
    _maskedImage.setXY0(origin);
}


// Write FITS

template<typename ImageT, typename MaskT, typename VarianceT>
void afwImage::Exposure<ImageT, MaskT, VarianceT>::writeFits(std::string const & fileName) const {
    fits::Fits fitsfile(fileName, "w", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile);
}

template<typename ImageT, typename MaskT, typename VarianceT>
void afwImage::Exposure<ImageT, MaskT, VarianceT>::writeFits(fits::MemFileManager & manager) const {
    fits::Fits fitsfile(manager, "w", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile);
}

template<typename ImageT, typename MaskT, typename VarianceT>
void afwImage::Exposure<ImageT, MaskT, VarianceT>::writeFits(fits::Fits & fitsfile) const {
    ExposureInfo::FitsWriteData data = _info->_startWriteFits(getXY0());
    _maskedImage.writeFits(
        fitsfile, data.metadata,
        data.imageMetadata, data.maskMetadata, data.varianceMetadata
    );
    _info->_finishWriteFits(fitsfile, data);
}

// Explicit instantiations
/// @cond
template class afwImage::Exposure<std::uint16_t>;
template class afwImage::Exposure<int>;
template class afwImage::Exposure<float>;
template class afwImage::Exposure<double>;
template class afwImage::Exposure<std::uint64_t>;
/// @endcond
