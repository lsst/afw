/*
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "lsst/log/Log.h"
#include "boost/algorithm/string/trim.hpp"

#include "lsst/afw/image/MaskedImageFitsReader.h"

namespace lsst { namespace afw { namespace image {

namespace {

fits::Fits * nextHdu(fits::Fits * fitsFile) {
    if (fitsFile == nullptr) {
        return nullptr;
    }
    try {
        fitsFile->setHdu(1, true);
    } catch (fits::FitsError &) {
        fitsFile->status = 0;
        return nullptr;
    }
    return fitsFile;
}

} // anonymous

MaskedImageFitsReader::MaskedImageFitsReader(std::string const& fileName, int hdu) :
    _imageReader(fileName, hdu),
    _maskReader(nextHdu(_imageReader._fitsFile)),
    _varianceReader(nextHdu(_maskReader._fitsFile))
{}

MaskedImageFitsReader::MaskedImageFitsReader(fits::MemFileManager& manager, int hdu) :
    _imageReader(manager, hdu),
    _maskReader(nextHdu(_imageReader._fitsFile)),
    _varianceReader(nextHdu(_maskReader._fitsFile))
{}

MaskedImageFitsReader::MaskedImageFitsReader(fits::Fits * fitsFile) :
    _imageReader(fitsFile),
    _maskReader(nextHdu(_imageReader._fitsFile)),
    _varianceReader(nextHdu(_maskReader._fitsFile))
{}

MaskedImageFitsReader::~MaskedImageFitsReader() noexcept = default;

std::string MaskedImageFitsReader::readImageDType() const { return _imageReader.readDType(); }

std::string MaskedImageFitsReader::readMaskDType() const { return _maskReader.readDType(); }

std::string MaskedImageFitsReader::readVarianceDType() const { return _varianceReader.readDType(); }

lsst::geom::Box2I MaskedImageFitsReader::readBBox(ImageOrigin origin) {
    return _imageReader.readBBox(origin);
}

lsst::geom::Point2I MaskedImageFitsReader::readXY0(lsst::geom::Box2I const & bbox, ImageOrigin origin) {
    return _imageReader.readXY0(bbox, origin);
}

namespace {

void checkExtType(ImageBaseFitsReader const & reader, std::shared_ptr<daf::base::PropertyList> metadata,
                  std::string const& expected) {
    try {
        std::string exttype = boost::algorithm::trim_right_copy(metadata->getAsString("EXTTYPE"));
        if (exttype != "" && exttype != expected) {
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                              str(boost::format("Reading %s (hdu %d) Expected EXTTYPE==\"%s\", saw \"%s\"") %
                               reader.getFileName() % reader.getHdu() % expected % exttype));
        }
        metadata->remove("EXTTYPE");
    } catch (pex::exceptions::NotFoundError) {
        LOGL_WARN("afw.image.MaskedImageFitsReader", "Expected extension type not found: %s",
                  expected.c_str());
    }
}

} // anonymous

std::shared_ptr<daf::base::PropertyList> MaskedImageFitsReader::readPrimaryMetadata() {
    auto fitsFile = _imageReader._fitsFile;
    fits::HduMoveGuard guard(*fitsFile, 0);
    return fits::readMetadata(*fitsFile, /*strip=*/true);
}

std::shared_ptr<daf::base::PropertyList> MaskedImageFitsReader::readImageMetadata() {
    if (!_imageMetadata) {
        _imageMetadata = _imageReader.readMetadata();
        checkExtType(_imageReader, _imageMetadata, "IMAGE");
    }
    return _imageMetadata;
}

std::shared_ptr<daf::base::PropertyList> MaskedImageFitsReader::readMaskMetadata() {
    if (!_maskMetadata) {
        _maskReader.readMetadata();
        checkExtType(_maskReader, _maskMetadata, "MASK");
    }
    return _maskMetadata;
}

std::shared_ptr<daf::base::PropertyList> MaskedImageFitsReader::readVarianceMetadata() {
    if (!_varianceMetadata) {
        _varianceReader.readMetadata();
        checkExtType(_varianceReader, _varianceMetadata, "VARIANCE");
    }
    return _varianceMetadata;
}

template <typename ImagePixelT>
Image<ImagePixelT> MaskedImageFitsReader::readImage(lsst::geom::Box2I const & bbox, ImageOrigin origin,
                                                    bool allowUnsafe) {
    return _imageReader.read<ImagePixelT>(bbox, origin, allowUnsafe);
}

template <typename MaskPixelT>
Mask<MaskPixelT> MaskedImageFitsReader::readMask(lsst::geom::Box2I const & bbox, ImageOrigin origin,
                                                 bool conformMasks, bool allowUnsafe) {
    return _maskReader.read<MaskPixelT>(bbox, origin, conformMasks, allowUnsafe);
}

template <typename VariancePixelT>
Image<VariancePixelT> MaskedImageFitsReader::readVariance(lsst::geom::Box2I const & bbox,
                                                          ImageOrigin origin, bool allowUnsafe) {
    return _varianceReader.read<VariancePixelT>(bbox, origin, allowUnsafe);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT> MaskedImageFitsReader::read(
    lsst::geom::Box2I const & bbox, ImageOrigin origin,
    bool conformMasks, bool needAllHdus, bool allowUnsafe
) {
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

    LOG_LOGGER _log = LOG_GET("afw.image.MaskedImageFitsReader");

    enum class Hdu { Primary = 0, Image, Mask, Variance };

    // If the user has requested a non-default HDU and we require all HDUs, we fail.
    if (needAllHdus && _imageReader.getHdu() > static_cast<int>(Hdu::Image)) {
        throw LSST_EXCEPT(fits::FitsError, "Cannot read all HDUs starting from non-default");
    }

    auto image = std::make_shared<Image<ImagePixelT>>(_imageReader.read<ImagePixelT>(bbox, origin,
                                                                                     allowUnsafe));
    std::shared_ptr<Mask<MaskPixelT>> mask;
    std::shared_ptr<Image<VariancePixelT>> variance;

    // only read other planes if they're in their normal HDUs
    if (_imageReader.getHdu() == static_cast<int>(Hdu::Image)) {
        try {
            mask = std::make_shared<Mask<MaskPixelT>>(
                _maskReader.read<MaskPixelT>(bbox, origin, conformMasks, allowUnsafe)
            );
        } catch (fits::FitsError& e) {
            if (needAllHdus) {
                LSST_EXCEPT_ADD(e, "Reading Mask");
                throw e;
            }
            LOGL_WARN(_log, "Mask unreadable (%s); using default", e.what());
            // By resetting the status we are able to read the next HDU (the variance).
            _maskReader._fitsFile->status = 0;
        }
        try {
            variance = std::make_shared<Image<VariancePixelT>>(
                _varianceReader.read<VariancePixelT>(bbox, origin, allowUnsafe)
            );
        } catch (fits::FitsError& e) {
            if (needAllHdus) {
                LSST_EXCEPT_ADD(e, "Reading Variance");
                throw e;
            }
            LOGL_WARN(_log, "Variance unreadable (%s); using default", e.what());
            // By resetting the status we are able to read the next HDU (the variance).
            _varianceReader._fitsFile->status = 0;
        }
    }
    return MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>(image, mask, variance);
}

#define INSTANTIATE(ImagePixelT) \
    template MaskedImage<ImagePixelT, MaskPixel, VariancePixel> MaskedImageFitsReader::read( \
        lsst::geom::Box2I const &, \
        ImageOrigin, \
        bool, bool, bool \
    ); \
    template Image<ImagePixelT> MaskedImageFitsReader::readImage(\
        lsst::geom::Box2I const &, \
        ImageOrigin, \
        bool \
    )

INSTANTIATE(std::uint16_t);
INSTANTIATE(int);
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(std::uint64_t);

template Mask<MaskPixel> MaskedImageFitsReader::readMask(lsst::geom::Box2I const &, ImageOrigin, bool, bool);
template Image<VariancePixel> MaskedImageFitsReader::readVariance(lsst::geom::Box2I const &, ImageOrigin,
                                                                  bool);

}}} // lsst::afw::image
