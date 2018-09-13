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

#include "lsst/afw/image/ImageBaseFitsReader.h"
#include "lsst/afw/geom/wcsUtils.h"

namespace lsst { namespace afw { namespace image {

ImageBaseFitsReader::ImageBaseFitsReader(std::string const& fileName, int hdu) :
    _ownsFitsFile(true),
    _hdu(0),
    _fitsFile(new fits::Fits(fileName, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK))
{
    _fitsFile->setHdu(hdu);
    _fitsFile->checkCompressedImagePhu();
    _hdu = _fitsFile->getHdu();
}

ImageBaseFitsReader::ImageBaseFitsReader(fits::MemFileManager& manager, int hdu) :
    _ownsFitsFile(true),
    _hdu(0),
    _fitsFile(new fits::Fits(manager, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK))
{
    _fitsFile->setHdu(hdu);
    _fitsFile->checkCompressedImagePhu();
    _hdu = _fitsFile->getHdu();
}

ImageBaseFitsReader::ImageBaseFitsReader(fits::Fits * fitsFile) :
    _ownsFitsFile(false),
    _hdu(0),
    _fitsFile(fitsFile)
{
    if (_fitsFile) {
        if (_fitsFile->getHdu() == 0 && _fitsFile->getImageDim() == 0) {
            _fitsFile->setHdu(fits::DEFAULT_HDU);
        }
        _fitsFile->checkCompressedImagePhu();
        _hdu = _fitsFile->getHdu();
    }
}

ImageBaseFitsReader::~ImageBaseFitsReader() noexcept {
    if (_ownsFitsFile) {
        delete _fitsFile;
    }
}

namespace {

void checkFitsFile(fits::Fits * f) {
    if (f == nullptr) {
        throw LSST_EXCEPT(
            fits::FitsError,
            "FitsReader not initialized; desired HDU is probably missing."
        );
    }
}

} // anonymous

std::string ImageBaseFitsReader::readDType() const {
    checkFitsFile(_fitsFile);
    fits::HduMoveGuard guard(*_fitsFile, _hdu);
    return _fitsFile->getImageDType();
}

lsst::geom::Box2I ImageBaseFitsReader::readBBox(ImageOrigin origin) {
    readMetadata();  // guarantees _bbox is initialized
    if (origin == LOCAL) {
        return lsst::geom::Box2I(lsst::geom::Point2I(), _bbox.getDimensions());
    }
    return _bbox;
}

lsst::geom::Point2I ImageBaseFitsReader::readXY0(lsst::geom::Box2I const & bbox, ImageOrigin origin) {
    if (bbox.isEmpty()) {
        return readBBox().getMin();
    } else if (origin == PARENT) {
        return bbox.getMin();
    } else {
        auto full = readBBox();
        return full.getMin() + lsst::geom::Extent2I(bbox.getMin());
    }
}

std::shared_ptr<daf::base::PropertyList> ImageBaseFitsReader::readMetadata() {
    checkFitsFile(_fitsFile);
    if (_metadata == nullptr) {
        fits::HduMoveGuard guard(*_fitsFile, _hdu);
        auto metadata = fits::readMetadata(*_fitsFile, /*strip=*/true);
        // One-use lambda avoids declare-then-initialize; see C++ Core Guidelines, ES28.
        auto computeShape = [this](){
            int nAxis = _fitsFile->getImageDim();
            if (nAxis == 2) {
                return _fitsFile->getImageShape<2>();
            }
            if (nAxis == 3) {
                ndarray::Vector<ndarray::Size, 3> shape3 = _fitsFile->getImageShape<3>();
                if (shape3[0] != 1) {
                    throw LSST_EXCEPT(
                        fits::FitsError,
                        str(boost::format("Error reading %s: HDU %d has 3rd dimension %d != 1") %
                            getFileName() % _hdu % shape3[0]));
                }
                return shape3.last<2>();
            }
            throw LSST_EXCEPT(
                fits::FitsError,
                str(boost::format("Error reading %s: HDU %d has %d dimensions") %
                    getFileName() % _hdu % nAxis)
            );
        };
        // Construct _bbox and check dimensions at the same time both to make
        // one less valid state for this class to be in and to make sure
        // the appropriate metadata is stripped before we ever return it.
        ndarray::Vector<ndarray::Size, 2> shape = computeShape();
        auto xy0 = afw::geom::getImageXY0FromMetadata(*metadata, detail::wcsNameForXY0, /*strip=*/true);
        _bbox = lsst::geom::Box2I(xy0, lsst::geom::Extent2I(shape[1], shape[0]));
        _metadata = std::move(metadata);
    }
    return _metadata;
}

namespace {

// Return a numpy-like description of a primitive type (e.g. uint8, float64).
template <typename T>
std::string makeDTypeString() {
    using L = std::numeric_limits<T>;
    static_assert(L::is_specialized, "makeDTypeString requires a primitive numeric template parameter.");
    std::string result;
    if (std::is_same<T, bool>::value) {
        result += "bool";
    } else {
        if (L::is_integer) {
            if (L::is_signed) {
                result += "int";
            } else {
                result += "uint";
            }
        } else {
            result += "float";
        }
        result += std::to_string(sizeof(T)*8);
    }
    return result;
}

} // anonymous

template <typename T>
ndarray::Array<T, 2, 2> ImageBaseFitsReader::readArray(lsst::geom::Box2I const & bbox, ImageOrigin origin) {
    checkFitsFile(_fitsFile);
    auto fullBBox = readBBox(origin);
    auto subBBox = bbox;
    if (subBBox.isEmpty()) {
        subBBox = fullBBox;
    } else if (!fullBBox.contains(subBBox)) {
        throw LSST_EXCEPT(
            pex::exceptions::LengthError,
            str(boost::format("Subimage box (%d,%d) %dx%d doesn't fit in image (%d,%d) %dx%d in HDU %d") %
                subBBox.getMinX() % subBBox.getMinY() % subBBox.getWidth() % subBBox.getHeight() %
                fullBBox.getMinX() % fullBBox.getMinY() % fullBBox.getWidth() % fullBBox.getHeight() % _hdu)
        );
    }
    fits::HduMoveGuard guard(*_fitsFile, _hdu);
    if (!_fitsFile->checkImageType<T>()) {
        throw LSST_FITS_EXCEPT(
            fits::FitsTypeError, *_fitsFile,
            str(boost::format("Incompatible type for FITS image: on disk is %s (HDU %d), in-memory is %s.") %
                _fitsFile->getImageDType() % _hdu % makeDTypeString<T>())
        );
    }
    ndarray::Array<T, 2, 2> result = ndarray::allocate(subBBox.getHeight(), subBBox.getWidth());
    ndarray::Vector<int, 2> offset = ndarray::makeVector(subBBox.getMinY() - fullBBox.getMinY(),
                                                         subBBox.getMinX() - fullBBox.getMinX());
    _fitsFile->readImage(result, offset);
    return result;
}


#define INSTANTIATE(T) \
    template ndarray::Array<T, 2, 2> ImageBaseFitsReader::readArray( \
        lsst::geom::Box2I const & bbox, \
        ImageOrigin origin \
    )

INSTANTIATE(std::uint16_t);
INSTANTIATE(int);
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(std::uint64_t);

}}} // lsst::afw::image
