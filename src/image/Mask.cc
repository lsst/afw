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
 * The fundamental type defined here is Mask; a 2-d array of pixels.  Each of
 * these pixels should be thought of as a set of bits, and the names of these
 * bits are given by MaskPlaneDict (which is implemented as a std::map)
 */

#include <functional>
#include <list>
#include <string>
#include <sstream>
#include "boost/format.hpp"

#include "lsst/daf/base.h"
#include "lsst/geom.h"
#include "lsst/pex/exceptions.h"
#include "lsst/log/Log.h"
#include "lsst/afw/image/Mask.h"
#include "lsst/afw/image/LsstImageTypes.h"
#include "lsst/afw/image/detail/MaskDict.h"
#include "lsst/afw/image/MaskFitsReader.h"

namespace dafBase = lsst::daf::base;
namespace pexExcept = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace image {

// deprecated on DM-32438
template <typename MaskPixelT>
void Mask<MaskPixelT>::_initializePlanes(MaskPlaneDict const& planeDefs) {
    LOGL_DEBUG("lsst.afw.image.Mask", "Number of possible mask planes: %d", getNumPlanesMax());

    LOGL_WARN("lsst.afw.image.Mask",
              "Replaced by a shared_ptr interface to MaskDict; does not handle docstrings. Will be removed "
              "after v26.");
}

// deprecated on DM-32438
template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(unsigned int width, unsigned int height, MaskPlaneDict const& planeDefs)
        : ImageBase<MaskPixelT>(lsst::geom::ExtentI(width, height)),
          _maskDict(detail::MaskDict(getNumPlanesMax(), planeDefs, detail::MaskPlaneDocDict())) {
    *this = 0x0;
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(unsigned int width, unsigned int height, std::optional<detail::MaskDict> maskDict)
        : ImageBase<MaskPixelT>(lsst::geom::ExtentI(width, height)),
          _maskDict(maskDict.value_or(detail::MaskDict(getNumPlanesMax()))) {
    *this = 0x0;
}

// deprecated on DM-32438
template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(unsigned int width, unsigned int height, MaskPixelT initialValue,
                       MaskPlaneDict const& planeDefs)
        : ImageBase<MaskPixelT>(lsst::geom::ExtentI(width, height)),
          _maskDict(detail::MaskDict(getNumPlanesMax(), planeDefs, detail::MaskPlaneDocDict())) {
    _initializePlanes(planeDefs);
    *this = initialValue;
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(unsigned int width, unsigned int height, MaskPixelT initialValue,
                       std::optional<detail::MaskDict> maskDict)
        : ImageBase<MaskPixelT>(lsst::geom::ExtentI(width, height)),
          _maskDict(maskDict.value_or(detail::MaskDict(getNumPlanesMax()))) {
    *this = initialValue;
}

// deprecated on DM-32438
template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(lsst::geom::Extent2I const& dimensions, MaskPlaneDict const& planeDefs)
        : ImageBase<MaskPixelT>(dimensions),
          _maskDict(detail::MaskDict(getNumPlanesMax(), planeDefs, detail::MaskPlaneDocDict())) {
    _initializePlanes(planeDefs);
    *this = 0x0;
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(lsst::geom::Extent2I const& dimensions, std::optional<detail::MaskDict> maskDict)
        : ImageBase<MaskPixelT>(dimensions),
          _maskDict(maskDict.value_or(detail::MaskDict(getNumPlanesMax()))) {
    *this = 0x0;
}

// deprecated on DM-32438
template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(lsst::geom::Extent2I const& dimensions, MaskPixelT initialValue,
                       MaskPlaneDict const& planeDefs)
        : ImageBase<MaskPixelT>(dimensions),
          _maskDict(detail::MaskDict(getNumPlanesMax(), planeDefs, detail::MaskPlaneDocDict())) {
    _initializePlanes(planeDefs);
    *this = initialValue;
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(lsst::geom::Extent2I const& dimensions, MaskPixelT initialValue,
                       std::optional<detail::MaskDict> maskDict)
        : ImageBase<MaskPixelT>(dimensions),
          _maskDict(maskDict.value_or(detail::MaskDict(getNumPlanesMax()))) {
    *this = initialValue;
}

// deprecated on DM-32438
template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(lsst::geom::Box2I const& bbox, MaskPlaneDict const& planeDefs)
        : ImageBase<MaskPixelT>(bbox),
          _maskDict(detail::MaskDict(getNumPlanesMax(), planeDefs, detail::MaskPlaneDocDict())) {
    _initializePlanes(planeDefs);
    *this = 0x0;
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(lsst::geom::Box2I const& bbox, std::optional<detail::MaskDict> maskDict)
        : ImageBase<MaskPixelT>(bbox), _maskDict(maskDict.value_or(detail::MaskDict(getNumPlanesMax()))) {
    *this = 0x0;
}

// deprecated on DM-32438
template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(lsst::geom::Box2I const& bbox, MaskPixelT initialValue, MaskPlaneDict const& planeDefs)
        : ImageBase<MaskPixelT>(bbox),
          _maskDict(detail::MaskDict(getNumPlanesMax(), planeDefs, detail::MaskPlaneDocDict())) {
    _initializePlanes(planeDefs);
    *this = initialValue;
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(lsst::geom::Box2I const& bbox, MaskPixelT initialValue,
                       std::optional<detail::MaskDict> maskDict)
        : ImageBase<MaskPixelT>(bbox), _maskDict(maskDict.value_or(detail::MaskDict(getNumPlanesMax()))) {
    *this = initialValue;
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(Mask const& rhs, lsst::geom::Box2I const& bbox, ImageOrigin const origin,
                       bool const deep)
        : ImageBase<MaskPixelT>(rhs, bbox, origin, deep), _maskDict(rhs._maskDict) {}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(Mask const& rhs, bool deep)
        : ImageBase<MaskPixelT>(rhs, deep), _maskDict(rhs._maskDict) {}
// Delegate to copy-constructor for backwards compatibility
template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(Mask&& rhs) : Mask(rhs, false) {}

template <typename MaskPixelT>
Mask<MaskPixelT>::~Mask() = default;

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(ndarray::Array<MaskPixelT, 2, 1> const& array, bool deep,
                       lsst::geom::Point2I const& xy0, std::optional<detail::MaskDict> maskDict)
        : image::ImageBase<MaskPixelT>(array, deep, xy0),
          _maskDict(maskDict.value_or(detail::MaskDict(getNumPlanesMax()))) {}

template <typename PixelT>
void Mask<PixelT>::swap(Mask& rhs) {
    using std::swap;  // See Meyers, Effective C++, Item 25

    ImageBase<PixelT>::swap(rhs);
    swap(_maskDict, rhs._maskDict);
}

template <typename PixelT>
void swap(Mask<PixelT>& a, Mask<PixelT>& b) {
    a.swap(b);
}

template <typename MaskPixelT>
Mask<MaskPixelT>& Mask<MaskPixelT>::operator=(const Mask<MaskPixelT>& rhs) {
    Mask tmp(rhs);
    swap(tmp);  // See Meyers, Effective C++, Item 11

    return *this;
}
// Delegate to copy-assignment for backwards compatibility
template <typename MaskPixelT>
Mask<MaskPixelT>& Mask<MaskPixelT>::operator=(Mask<MaskPixelT>&& rhs) {
    return *this = rhs;
}

template <typename MaskPixelT>
Mask<MaskPixelT>& Mask<MaskPixelT>::operator=(MaskPixelT const rhs) {
    fill_pixels(_getRawView(), rhs);

    return *this;
}

#ifndef DOXYGEN  // doc for this section is already in header

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(std::string const& fileName, int hdu, std::shared_ptr<daf::base::PropertySet> metadata,
                       lsst::geom::Box2I const& bbox, ImageOrigin origin, bool conformMasks, bool allowUnsafe)
        : ImageBase<MaskPixelT>(), _maskDict(getNumPlanesMax()) {
    MaskFitsReader reader(fileName, hdu);
    *this = reader.read<MaskPixelT>(bbox, origin, conformMasks, allowUnsafe);
    if (metadata) {
        metadata->combine(*reader.readMetadata());
    }
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(fits::MemFileManager& manager, int hdu,
                       std::shared_ptr<daf::base::PropertySet> metadata, lsst::geom::Box2I const& bbox,
                       ImageOrigin origin, bool conformMasks, bool allowUnsafe)
        : ImageBase<MaskPixelT>(), _maskDict(getNumPlanesMax()) {
    MaskFitsReader reader(manager, hdu);
    *this = reader.read<MaskPixelT>(bbox, origin, conformMasks, allowUnsafe);
    if (metadata) {
        metadata->combine(*reader.readMetadata());
    }
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(fits::Fits& fitsFile, std::shared_ptr<daf::base::PropertySet> metadata,
                       lsst::geom::Box2I const& bbox, ImageOrigin const origin, bool const conformMasks,
                       bool allowUnsafe)
        : ImageBase<MaskPixelT>(), _maskDict(getNumPlanesMax()) {
    MaskFitsReader reader(&fitsFile);
    *this = reader.read<MaskPixelT>(bbox, origin, conformMasks, allowUnsafe);
    if (metadata) {
        metadata->combine(*reader.readMetadata());
    }
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::writeFits(std::string const& fileName, daf::base::PropertySet const* metadata_i,
                                 std::string const& mode) const {
    fits::Fits fitsfile(fileName, mode, fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, metadata_i);
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::writeFits(fits::MemFileManager& manager, daf::base::PropertySet const* metadata_i,
                                 std::string const& mode) const {
    fits::Fits fitsfile(manager, mode, fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, metadata_i);
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::writeFits(fits::Fits& fitsfile, daf::base::PropertySet const* metadata) const {
    writeFits(fitsfile, fits::ImageWriteOptions(*this), metadata);
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::writeFits(std::string const& filename, fits::ImageWriteOptions const& options,
                                 std::string const& mode, daf::base::PropertySet const* header) const {
    fits::Fits fitsfile(filename, mode, fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, options, header);
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::writeFits(fits::MemFileManager& manager, fits::ImageWriteOptions const& options,
                                 std::string const& mode, daf::base::PropertySet const* header) const {
    fits::Fits fitsfile(manager, mode, fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, options, header);
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::writeFits(fits::Fits& fitsfile, fits::ImageWriteOptions const& options,
                                 daf::base::PropertySet const* header) const {
    std::shared_ptr<daf::base::PropertySet> useHeader =
            header ? header->deepCopy() : std::make_shared<dafBase::PropertySet>();
    addMaskPlanesToMetadata(useHeader);
    fitsfile.writeImage(*this, options, useHeader.get());
}

#endif  // !DOXYGEN

template <typename MaskPixelT>
std::string Mask<MaskPixelT>::interpret(MaskPixelT value) {
    std::string result = "";
    for (auto const& pair : _maskDict.getMaskPlaneDict()) {
        if (value & getBitMaskFromPlaneId(pair.second)) {
            if (!result.empty()) {
                result += ",";
            }
            result += pair.first;
        }
    }
    return result;
}

// NOTE: static
// template <typename MaskPixelT>
// int Mask<MaskPixelT>::addMaskPlane(const std::string& name) {
//     return addMaskPlane(name, "");
// }

// NOTE: static
// template <typename MaskPixelT>
// int Mask<MaskPixelT>::addMaskPlane(const std::string& name, const std::string& doc) {
//     auto [id, newMaskDict] = detail::MaskDict::getDefault()->withNewMaskPlane(name, doc,
//     getNumPlanesMax()); detail::MaskDict::setDefault(newMaskDict); return id;
// }

// NOTE: static
// template <typename MaskPixelT>
// void Mask<MaskPixelT>::removeMaskPlane(const std::string& name) {
//     detail::MaskDict::setDefault(detail::MaskDict::getDefault()->withRemovedMaskPlane(name));
// }

template <typename MaskPixelT>
int Mask<MaskPixelT>::addPlane(const std::string& name, const std::string& doc) {
    return _maskDict.add(name, doc);
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::removeAndClearMaskPlane(const std::string& name) {
    clearMaskPlane(getPlaneId(name));  // clear this bit in this Mask
    _maskDict.remove(name);
}

// template <typename MaskPixelT>
// MaskPixelT Mask<MaskPixelT>::getBitMask(int planeId) {
//     MaskPlaneDict const& mpd = _defaultMaskDict()->getMaskPlaneDict();

//     for (auto const& i : mpd) {
//         if (planeId == i.second) {
//             MaskPixelT const bitmask = getBitMaskNoThrow(planeId);
//             if (bitmask == 0) {  // failed
//                 break;
//             }
//             return bitmask;
//         }
//     }
//     throw LSST_EXCEPT(pexExcept::InvalidParameterError,
//                       str(boost::format("Invalid mask plane ID: %d") % planeId));
// }

// NOTE: static
// template <typename MaskPixelT>
// int Mask<MaskPixelT>::getMaskPlane(const std::string& name) {
//     int const plane = detail::MaskDict()->getPlaneId(name);

//     if (plane < 0) {
//         throw LSST_EXCEPT(pexExcept::InvalidParameterError,
//                           str(boost::format("Invalid mask plane name: %s") % name));
//     } else {
//         return plane;
//     }
// }

template <typename MaskPixelT>
int Mask<MaskPixelT>::getPlaneId(std::string name) const {
    int plane = _maskDict.getPlaneId(name);

    if (plane < 0) {
        std::string planeStr = _maskDict.print();
        throw LSST_EXCEPT(
                pexExcept::InvalidParameterError,
                str(boost::format("Invalid mask plane name: '%s'. Known planes:\n%s") % name % planeStr));
    }
    return plane;
}

// NOTE: static
// template <typename MaskPixelT>
// MaskPixelT Mask<MaskPixelT>::getPlaneBitMask(const std::string& name) {
//     return getBitMaskFromPlaneId(getMaskPlane(name));
// }

template <typename MaskPixelT>
MaskPixelT Mask<MaskPixelT>::getBitMask(std::string name) const {
    return getBitMaskFromPlaneId(getPlaneId(name));
}

// NOTE: static
// template <typename MaskPixelT>
// MaskPixelT Mask<MaskPixelT>::getPlaneBitMask(const std::vector<std::string>& names) {
//     MaskPixelT mpix = 0x0;
//     for (auto const& name : names) {
//         mpix |= getBitMaskFromPlaneId(getMaskPlane(name));
//     }
//     return mpix;
// }

template <typename MaskPixelT>
MaskPixelT Mask<MaskPixelT>::getBitMask(const std::vector<std::string>& names) const {
    MaskPixelT mpix = 0x0;
    for (auto const& name : names) {
        mpix |= getBitMaskFromPlaneId(getPlaneId(name));
    }
    return mpix;
}

template <typename MaskPixelT>
int Mask<MaskPixelT>::getNumPlanesUsed() {
    return _maskDict.getMaskPlaneDict().size();
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::clearAllMaskPlanes() {
    *this = 0;
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::clearMaskPlane(int planeId) {
    *this &= ~getBitMaskFromPlaneId(planeId);
}

// NOTE: static
// template <typename MaskPixelT>
// void Mask<MaskPixelT>::clearDefaultMaskDict() {
//     detail::MaskDict::clearDefaultPlanes();
// }

// NOTE: static
// template <typename MaskPixelT>
// void Mask<MaskPixelT>::restoreDefaultMaskDict() {
//     detail::MaskDict::restoreDefaultMaskDict();
// }

// NOTE: static
// template <typename MaskPixelT>
// void Mask<MaskPixelT>::setDefaultMaskDict(MaskDict maskDict) {
//     detail::MaskDict::setDefault(maskDict);
// }

template <typename MaskPixelT>
void Mask<MaskPixelT>::conformMaskPlanes(detail::MaskDict const currentMaskDict) {
    // std::shared_ptr<detail::MaskDict> currentMD = detail::MaskDict::copyOrGetDefault(currentMaskDict);
    // auto currentMD = detail::MaskDict::getDefaultIfEmpty(currentMaskDict);

    std::cout << "\n_maskDict\n";
    _maskDict.print();
    // std::cout << "\ncurrentMD\n";
    // currentMD.print();
    std::cout << "\ndefault\n";
    detail::MaskDict(getNumPlanesMax()).print();
    std::cout << "\n";
    // bool returnDefault = true;
    // TODO: rewrite this whole thing!
    if (_maskDict == currentMaskDict) {
        // Are we still the default?
        if (detail::MaskDict(getNumPlanesMax()) == _maskDict) {
            return;  // nothing to do
        }
    } else {
        // Find out which planes need to be permuted
        MaskPixelT keepBitmask = 0;                        // mask of bits to keep
        MaskPixelT canonicalMask[sizeof(MaskPixelT) * 8];  // bits in lsst::afw::image::Mask that should be
        MaskPixelT currentMask[sizeof(MaskPixelT) * 8];    //           mapped to these bits
        int numReMap = 0;

        for (auto const& i : currentMaskDict.getMaskPlaneDict()) {
            std::string const name = i.first;         // name of mask plane
            int const currentPlaneNumber = i.second;  // plane number currently in use
            // Default to an empty docstring, for forwards compatibility.
            std::string currentPlaneDoc = "";
            // TODO: can we make the docs defaulting to empty be a class invariant?
            if (currentMaskDict.getMaskPlaneDocDict().find(name) !=
                currentMaskDict.getMaskPlaneDocDict().end())
                currentPlaneDoc = currentMaskDict.getMaskPlaneDocDict().at(name);
            std::cout << "> " << name << " " << currentPlaneNumber << " " << currentPlaneDoc << std::endl;
            // int canonicalPlaneNumber = getPlaneId(name);  // plane number in lsst::afw::image::Mask

            // if (canonicalPlaneNumber < 0) {  // no such plane; add it
            // canonicalPlaneNumber = addMaskPlane(name, currentPlaneDoc);
            // }

            // if (canonicalPlaneNumber == currentPlaneNumber) {
            //     keepBitmask |=
            //             getBitMaskFromPlaneId(canonicalPlaneNumber);  // bit is unchanged, so preserve it
            // } else {
            //     canonicalMask[numReMap] = getBitMaskFromPlaneId(canonicalPlaneNumber);
            //     currentMask[numReMap] = getBitMaskFromPlaneId(currentPlaneNumber);
            //     numReMap++;
            // }

            // canonicalPlaneDoc = ??
            // if (currentPlaneDoc != canonicalPlaneDoc) {
            //     if (canonicalPlaneDoc == "") { // use .empty?
            //         detail::MaskDict::getDefault()->setDoc(name, currentPlanedoc);
            //     }
            //     else if (currentPlaneDoc == "")
            //     {
            //         // ??
            //     }
            //     else {
            //         // detach the default
            //     }
            // }
        }

        // Now loop over all pixels in Mask
        if (numReMap > 0) {
            for (int r = 0; r != this->getHeight(); ++r) {  // "this->": Meyers, Effective C++, Item 43
                for (typename Mask::x_iterator ptr = this->row_begin(r), end = this->row_end(r); ptr != end;
                     ++ptr) {
                    MaskPixelT const pixel = *ptr;

                    MaskPixelT newPixel = pixel & keepBitmask;  // value of invariant mask bits
                    for (int i = 0; i < numReMap; i++) {
                        if (pixel & currentMask[i]) newPixel |= canonicalMask[i];
                    }

                    *ptr = newPixel;
                }
            }
        }
    }
    // We've made the planes match the current mask dictionary
    // _maskDict = detail::MaskDict::getDefault();
}

template <typename MaskPixelT>
typename ImageBase<MaskPixelT>::PixelReference Mask<MaskPixelT>::operator()(int x, int y) {
    return this->ImageBase<MaskPixelT>::operator()(x, y);
}

template <typename MaskPixelT>
typename ImageBase<MaskPixelT>::PixelReference Mask<MaskPixelT>::operator()(int x, int y,
                                                                            CheckIndices const& check) {
    return this->ImageBase<MaskPixelT>::operator()(x, y, check);
}

template <typename MaskPixelT>
typename ImageBase<MaskPixelT>::PixelConstReference Mask<MaskPixelT>::operator()(int x, int y) const {
    return this->ImageBase<MaskPixelT>::operator()(x, y);
}

template <typename MaskPixelT>
typename ImageBase<MaskPixelT>::PixelConstReference Mask<MaskPixelT>::operator()(
        int x, int y, CheckIndices const& check) const {
    return this->ImageBase<MaskPixelT>::operator()(x, y, check);
}

template <typename MaskPixelT>
bool Mask<MaskPixelT>::operator()(int x, int y, int planeId) const {
    // !! converts an int to a bool
    return !!(this->ImageBase<MaskPixelT>::operator()(x, y) & getBitMaskFromPlaneId(planeId));
}

template <typename MaskPixelT>
bool Mask<MaskPixelT>::operator()(int x, int y, int planeId, CheckIndices const& check) const {
    // !! converts an int to a bool
    return !!(this->ImageBase<MaskPixelT>::operator()(x, y, check) & getBitMaskFromPlaneId(planeId));
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::checkMaskDictionaries(Mask<MaskPixelT> const& other) {
    if (_maskDict != other._maskDict) {
        throw LSST_EXCEPT(pexExcept::RuntimeError, "Mask dictionaries do not match");
    }
}

template <typename MaskPixelT>
Mask<MaskPixelT>& Mask<MaskPixelT>::operator|=(MaskPixelT const val) {
    transform_pixels(_getRawView(), _getRawView(),
                     [&val](MaskPixelT const& l) -> MaskPixelT { return l | val; });
    return *this;
}

template <typename MaskPixelT>
Mask<MaskPixelT>& Mask<MaskPixelT>::operator|=(Mask const& rhs) {
    checkMaskDictionaries(rhs);

    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(pexExcept::LengthError,
                          str(boost::format("Images are of different size, %dx%d v %dx%d") %
                              this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()));
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(),
                     [](MaskPixelT const& l, MaskPixelT const& r) -> MaskPixelT { return l | r; });
    return *this;
}

template <typename MaskPixelT>
Mask<MaskPixelT>& Mask<MaskPixelT>::operator&=(MaskPixelT const val) {
    transform_pixels(_getRawView(), _getRawView(), [&val](MaskPixelT const& l) { return l & val; });
    return *this;
}

template <typename MaskPixelT>
Mask<MaskPixelT>& Mask<MaskPixelT>::operator&=(Mask const& rhs) {
    checkMaskDictionaries(rhs);

    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(pexExcept::LengthError,
                          str(boost::format("Images are of different size, %dx%d v %dx%d") %
                              this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()));
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(),
                     [](MaskPixelT const& l, MaskPixelT const& r) -> MaskPixelT { return l & r; });
    return *this;
}

template <typename MaskPixelT>
Mask<MaskPixelT>& Mask<MaskPixelT>::operator^=(MaskPixelT const val) {
    transform_pixels(_getRawView(), _getRawView(),
                     [&val](MaskPixelT const& l) -> MaskPixelT { return l ^ val; });
    return *this;
}

template <typename MaskPixelT>
Mask<MaskPixelT>& Mask<MaskPixelT>::operator^=(Mask const& rhs) {
    checkMaskDictionaries(rhs);

    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(pexExcept::LengthError,
                          str(boost::format("Images are of different size, %dx%d v %dx%d") %
                              this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()));
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(),
                     [](MaskPixelT const& l, MaskPixelT const& r) -> MaskPixelT { return l ^ r; });
    return *this;
}

// template <typename MaskPixelT>
// void Mask<MaskPixelT>::setMaskPlaneValues(int const planeId, int const x0, int const x1, int const y) {
//     MaskPixelT const bitMask = getBitMaskFromPlaneId(planeId);

//     for (int x = x0; x <= x1; x++) {
//         operator()(x, y) = operator()(x, y) | bitMask;
//     }
// }

template <typename MaskPixelT>
void Mask<MaskPixelT>::addMaskPlanesToMetadata(std::shared_ptr<dafBase::PropertySet> metadata) const {
    if (!metadata) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, "Null std::shared_ptr<PropertySet>");
    }

    // First, clear existing MaskPlane metadata
    using NameList = std::vector<std::string>;
    NameList paramNames = metadata->paramNames(false);
    for (auto const& paramName : paramNames) {
        if (paramName.compare(0, maskPlanePrefix.size(), maskPlanePrefix) == 0) {
            metadata->remove(paramName);
        }
    }

    MaskPlaneDict const& mpdict = _maskDict.getMaskPlaneDict();
    MaskPlaneDocDict const& mpdocs = _maskDict.getMaskPlaneDocDict();

    // Add new MaskPlane metadata
    for (auto const& i : mpdict) {
        std::string const& planeName = i.first;
        int const planeNumber = i.second;

        if (planeName != "") {
            metadata->add(maskPlanePrefix + planeName, planeNumber);
            try {
                metadata->add(maskPlaneDocPrefix + planeName, mpdocs.at(planeName));
            } catch (const std::out_of_range&) {
                // Write an empty docstring, for forwards compatibility.
                metadata->add(maskPlaneDocPrefix + planeName, "");
            }
        }
    }
}

template <typename MaskPixelT>
detail::MaskDict Mask<MaskPixelT>::parseMaskPlaneMetadata(
        std::shared_ptr<dafBase::PropertySet const> metadata) {
    MaskPlaneDict bitDict;
    MaskPlaneDocDict docDict;

    // First, clear existing MaskPlane metadata
    using NameList = std::vector<std::string>;
    NameList paramNames = metadata->paramNames(false);
    int numPlanesUsed = 0;  // number of planes used

    // Iterate over childless properties with names starting with maskPlanePrefix
    for (auto const& paramName : paramNames) {
        if (paramName.compare(0, maskPlanePrefix.size(), maskPlanePrefix) == 0) {
            // split off maskPlanePrefix to obtain plane name
            std::string planeName = paramName.substr(maskPlanePrefix.size());
            int const planeId = metadata->getAsInt(paramName);

            MaskPlaneDict::const_iterator plane = bitDict.find(planeName);
            if (plane != bitDict.end() && planeId != plane->second) {
                throw LSST_EXCEPT(pexExcept::RuntimeError, "File specifies plane " + planeName + " twice");
            }
            for (auto const& j : bitDict) {
                if (planeId == j.second) {
                    throw LSST_EXCEPT(pexExcept::RuntimeError,
                                      str(boost::format("File specifies plane %s has same value (%d) as %s") %
                                          planeName % planeId % j.first));
                }
            }
            // build new entry
            if (numPlanesUsed >= getNumPlanesMax()) {
                // Max number of planes already allocated
                throw LSST_EXCEPT(
                        pexExcept::RuntimeError,
                        str(boost::format("Max number of planes (%1%) already used") % getNumPlanesMax()));
            }
            bitDict[planeName] = planeId;
            // Backwards compatibility: mask planes are undocumented, unless a matching `MPD_` field is found.
            if (docDict.find(planeName) == docDict.end()) {
                docDict[planeName] = "";
            }
        } else if (paramName.compare(0, maskPlaneDocPrefix.size(), maskPlaneDocPrefix) == 0) {
            std::string planeName = paramName.substr(maskPlaneDocPrefix.size());
            std::string planeDoc = metadata->getAsString(paramName);
            // Overwrite the previous empty definition.
            docDict[planeName] = planeDoc;
        }
    }
    return detail::MaskDict(getNumPlanesMax(), bitDict, docDict);
}

template <typename MaskPixelT>
std::string Mask<MaskPixelT>::printMaskPlanes() const {
    return _maskDict.print();
}

/*
 * Static members of Mask
 */
template <typename MaskPixelT>
std::string const Mask<MaskPixelT>::maskPlanePrefix("MP_");
template <typename MaskPixelT>
std::string const Mask<MaskPixelT>::maskPlaneDocPrefix("MPD_");

//
// Explicit instantiations
//
template class Mask<MaskPixel>;
}  // namespace image
}  // namespace afw
}  // namespace lsst
