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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic pop
#include "boost/format.hpp"
#include "boost/filesystem/path.hpp"

#include "boost/functional/hash.hpp"

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

namespace {}  // namespace

template <typename MaskPixelT>
void Mask<MaskPixelT>::_initializePlanes(MaskPlaneDict const& planeDefs) {
    LOGL_DEBUG("afw.image.Mask", "Number of mask planes: %d", getNumPlanesMax());

    _maskDict = detail::MaskDict::copyOrGetDefault(planeDefs);
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(unsigned int width, unsigned int height, MaskPlaneDict const& planeDefs)
        : ImageBase<MaskPixelT>(lsst::geom::ExtentI(width, height)) {
    _initializePlanes(planeDefs);
    *this = 0x0;
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(unsigned int width, unsigned int height, MaskPixelT initialValue,
                       MaskPlaneDict const& planeDefs)
        : ImageBase<MaskPixelT>(lsst::geom::ExtentI(width, height)) {
    _initializePlanes(planeDefs);
    *this = initialValue;
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(lsst::geom::Extent2I const& dimensions, MaskPlaneDict const& planeDefs)
        : ImageBase<MaskPixelT>(dimensions) {
    _initializePlanes(planeDefs);
    *this = 0x0;
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(lsst::geom::Extent2I const& dimensions, MaskPixelT initialValue,
                       MaskPlaneDict const& planeDefs)
        : ImageBase<MaskPixelT>(dimensions) {
    _initializePlanes(planeDefs);
    *this = initialValue;
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(lsst::geom::Box2I const& bbox, MaskPlaneDict const& planeDefs)
        : ImageBase<MaskPixelT>(bbox) {
    _initializePlanes(planeDefs);
    *this = 0x0;
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(lsst::geom::Box2I const& bbox, MaskPixelT initialValue, MaskPlaneDict const& planeDefs)
        : ImageBase<MaskPixelT>(bbox) {
    _initializePlanes(planeDefs);
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
                       lsst::geom::Point2I const& xy0)
        : image::ImageBase<MaskPixelT>(array, deep, xy0), _maskDict(detail::MaskDict::getDefault()) {}

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
        : ImageBase<MaskPixelT>(), _maskDict(detail::MaskDict::getDefault()) {
    MaskFitsReader reader(fileName, hdu);
    *this = reader.read<MaskPixelT>(bbox, origin, conformMasks, allowUnsafe);
    if (metadata) {
        metadata->combine(reader.readMetadata());
    }
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(fits::MemFileManager& manager, int hdu,
                       std::shared_ptr<daf::base::PropertySet> metadata, lsst::geom::Box2I const& bbox,
                       ImageOrigin origin, bool conformMasks, bool allowUnsafe)
        : ImageBase<MaskPixelT>(), _maskDict(detail::MaskDict::getDefault()) {
    MaskFitsReader reader(manager, hdu);
    *this = reader.read<MaskPixelT>(bbox, origin, conformMasks, allowUnsafe);
    if (metadata) {
        metadata->combine(reader.readMetadata());
    }
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(fits::Fits& fitsFile, std::shared_ptr<daf::base::PropertySet> metadata,
                       lsst::geom::Box2I const& bbox, ImageOrigin const origin, bool const conformMasks,
                       bool allowUnsafe)
        : ImageBase<MaskPixelT>(), _maskDict(detail::MaskDict::getDefault()) {
    MaskFitsReader reader(&fitsFile);
    *this = reader.read<MaskPixelT>(bbox, origin, conformMasks, allowUnsafe);
    if (metadata) {
        metadata->combine(reader.readMetadata());
    }
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::writeFits(std::string const& fileName,
                                 std::shared_ptr<lsst::daf::base::PropertySet const> metadata_i,
                                 std::string const& mode) const {
    fits::Fits fitsfile(fileName, mode, fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, metadata_i);
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::writeFits(fits::MemFileManager& manager,
                                 std::shared_ptr<lsst::daf::base::PropertySet const> metadata_i,
                                 std::string const& mode) const {
    fits::Fits fitsfile(manager, mode, fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, metadata_i);
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::writeFits(fits::Fits& fitsfile,
                                 std::shared_ptr<lsst::daf::base::PropertySet const> metadata) const {
    writeFits(fitsfile, fits::ImageWriteOptions(*this), metadata);
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::writeFits(std::string const& filename, fits::ImageWriteOptions const& options,
                                 std::string const& mode,
                                 std::shared_ptr<daf::base::PropertySet const> header) const {
    fits::Fits fitsfile(filename, mode, fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, options, header);
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::writeFits(fits::MemFileManager& manager, fits::ImageWriteOptions const& options,
                                 std::string const& mode,
                                 std::shared_ptr<daf::base::PropertySet const> header) const {
    fits::Fits fitsfile(manager, mode, fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, options, header);
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::writeFits(fits::Fits& fitsfile, fits::ImageWriteOptions const& options,
                                 std::shared_ptr<daf::base::PropertySet const> header) const {
    std::shared_ptr<daf::base::PropertySet> useHeader =
            header ? header->deepCopy() : std::make_shared<dafBase::PropertySet>();
    addMaskPlanesToMetadata(useHeader);
    fitsfile.writeImage(*this, options, useHeader);
}

#endif  // !DOXYGEN

template <typename MaskPixelT>
std::string Mask<MaskPixelT>::interpret(MaskPixelT value) {
    std::string result = "";
    MaskPlaneDict const& mpd = _maskPlaneDict()->getMaskPlaneDict();
    for (MaskPlaneDict::const_iterator iter = mpd.begin(); iter != mpd.end(); ++iter) {
        if (value & getBitMask(iter->second)) {
            if (result.size() > 0) {
                result += ",";
            }
            result += iter->first;
        }
    }
    return result;
}

template <typename MaskPixelT>
int Mask<MaskPixelT>::addMaskPlane(const std::string& name) {
    int id = getMaskPlaneNoThrow(name);  // see if the plane is already available

    if (id < 0) {  // doesn't exist
        id = _maskPlaneDict()->getUnusedPlane();
    }

    // build new entry, adding the plane to all Masks where this is no contradiction

    if (id >= getNumPlanesMax()) {  // Max number of planes is already allocated
        throw LSST_EXCEPT(pexExcept::RuntimeError,
                          str(boost::format("Max number of planes (%1%) already used") % getNumPlanesMax()));
    }

    detail::MaskDict::addAllMasksPlane(name, id);

    return id;
}

template <typename MaskPixelT>
int Mask<MaskPixelT>::addMaskPlane(std::string name, int planeId) {
    if (planeId < 0 || planeId >= getNumPlanesMax()) {
        throw LSST_EXCEPT(
                pexExcept::RangeError,
                str(boost::format("mask plane ID must be between 0 and %1%") % (getNumPlanesMax() - 1)));
    }

    _maskPlaneDict()->add(name, planeId);

    return planeId;
}

template <typename MaskPixelT>
detail::MaskPlaneDict const& Mask<MaskPixelT>::getMaskPlaneDict() const {
    return _maskDict->getMaskPlaneDict();
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::removeMaskPlane(const std::string& name) {
    if (detail::MaskDict::getDefault()->getMaskPlane(name) < 0) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError,
                          str(boost::format("Plane %s doesn't exist in the default Mask") % name));
    }

    detail::MaskDict::detachDefault();  // leave current Masks alone
    _maskPlaneDict()->erase(name);
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::removeAndClearMaskPlane(const std::string& name, bool const removeFromDefault

) {
    clearMaskPlane(getMaskPlane(name));  // clear this bits in this Mask

    if (_maskDict == detail::MaskDict::getDefault() && removeFromDefault) {  // we are the default
        ;
    } else {
        _maskDict = _maskDict->clone();
    }

    _maskDict->erase(name);

    if (removeFromDefault && detail::MaskDict::getDefault()->getMaskPlane(name) >= 0) {
        removeMaskPlane(name);
    }
}

template <typename MaskPixelT>
MaskPixelT Mask<MaskPixelT>::getBitMaskNoThrow(int planeId) {
    return (planeId >= 0 && planeId < getNumPlanesMax()) ? (1 << planeId) : 0;
}

template <typename MaskPixelT>
MaskPixelT Mask<MaskPixelT>::getBitMask(int planeId) {
    MaskPlaneDict const& mpd = _maskPlaneDict()->getMaskPlaneDict();

    for (MaskPlaneDict::const_iterator i = mpd.begin(); i != mpd.end(); ++i) {
        if (planeId == i->second) {
            MaskPixelT const bitmask = getBitMaskNoThrow(planeId);
            if (bitmask == 0) {  // failed
                break;
            }
            return bitmask;
        }
    }
    throw LSST_EXCEPT(pexExcept::InvalidParameterError,
                      str(boost::format("Invalid mask plane ID: %d") % planeId));
}

template <typename MaskPixelT>
int Mask<MaskPixelT>::getMaskPlane(const std::string& name) {
    int const plane = getMaskPlaneNoThrow(name);

    if (plane < 0) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError,
                          str(boost::format("Invalid mask plane name: %s") % name));
    } else {
        return plane;
    }
}

template <typename MaskPixelT>
int Mask<MaskPixelT>::getMaskPlaneNoThrow(const std::string& name) {
    return _maskPlaneDict()->getMaskPlane(name);
}

template <typename MaskPixelT>
MaskPixelT Mask<MaskPixelT>::getPlaneBitMask(const std::string& name) {
    return getBitMask(getMaskPlane(name));
}

template <typename MaskPixelT>
MaskPixelT Mask<MaskPixelT>::getPlaneBitMask(const std::vector<std::string>& name) {
    MaskPixelT mpix = 0x0;
    for (std::vector<std::string>::const_iterator it = name.begin(); it != name.end(); ++it) {
        mpix |= getBitMask(getMaskPlane(*it));
    }
    return mpix;
}

template <typename MaskPixelT>
int Mask<MaskPixelT>::getNumPlanesUsed() {
    return _maskPlaneDict()->size();
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::clearMaskPlaneDict() {
    _maskPlaneDict()->clear();
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::clearAllMaskPlanes() {
    *this = 0;
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::clearMaskPlane(int planeId) {
    *this &= ~getBitMask(planeId);
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::conformMaskPlanes(MaskPlaneDict const& currentPlaneDict) {
    std::shared_ptr<detail::MaskDict> currentMD = detail::MaskDict::copyOrGetDefault(currentPlaneDict);

    if (*_maskDict == *currentMD) {
        if (*detail::MaskDict::getDefault() == *_maskDict) {
            return;  // nothing to do
        }
    } else {
        //
        // Find out which planes need to be permuted
        //
        MaskPixelT keepBitmask = 0;                        // mask of bits to keep
        MaskPixelT canonicalMask[sizeof(MaskPixelT) * 8];  // bits in lsst::afw::image::Mask that should be
        MaskPixelT currentMask[sizeof(MaskPixelT) * 8];    //           mapped to these bits
        int numReMap = 0;

        for (MaskPlaneDict::const_iterator i = currentPlaneDict.begin(); i != currentPlaneDict.end(); i++) {
            std::string const name = i->first;                     // name of mask plane
            int const currentPlaneNumber = i->second;              // plane number currently in use
            int canonicalPlaneNumber = getMaskPlaneNoThrow(name);  // plane number in lsst::afw::image::Mask

            if (canonicalPlaneNumber < 0) {  // no such plane; add it
                canonicalPlaneNumber = addMaskPlane(name);
            }

            if (canonicalPlaneNumber == currentPlaneNumber) {
                keepBitmask |= getBitMask(canonicalPlaneNumber);  // bit is unchanged, so preserve it
            } else {
                canonicalMask[numReMap] = getBitMask(canonicalPlaneNumber);
                currentMask[numReMap] = getBitMaskNoThrow(currentPlaneNumber);
                numReMap++;
            }
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
    _maskDict = detail::MaskDict::getDefault();
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
    return !!(this->ImageBase<MaskPixelT>::operator()(x, y) & getBitMask(planeId));
}

template <typename MaskPixelT>
bool Mask<MaskPixelT>::operator()(int x, int y, int planeId, CheckIndices const& check) const {
    // !! converts an int to a bool
    return !!(this->ImageBase<MaskPixelT>::operator()(x, y, check) & getBitMask(planeId));
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::checkMaskDictionaries(Mask<MaskPixelT> const& other) {
    if (*_maskDict != *other._maskDict) {
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

template <typename MaskPixelT>
void Mask<MaskPixelT>::setMaskPlaneValues(int const planeId, int const x0, int const x1, int const y) {
    MaskPixelT const bitMask = getBitMask(planeId);

    for (int x = x0; x <= x1; x++) {
        operator()(x, y) = operator()(x, y) | bitMask;
    }
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::addMaskPlanesToMetadata(std::shared_ptr<dafBase::PropertySet> metadata) {
    if (!metadata) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, "Null std::shared_ptr<PropertySet>");
    }

    // First, clear existing MaskPlane metadata
    typedef std::vector<std::string> NameList;
    NameList paramNames = metadata->paramNames(false);
    for (NameList::const_iterator i = paramNames.begin(); i != paramNames.end(); ++i) {
        if (i->compare(0, maskPlanePrefix.size(), maskPlanePrefix) == 0) {
            metadata->remove(*i);
        }
    }

    MaskPlaneDict const& mpd = _maskPlaneDict()->getMaskPlaneDict();

    // Add new MaskPlane metadata
    for (MaskPlaneDict::const_iterator i = mpd.begin(); i != mpd.end(); ++i) {
        std::string const& planeName = i->first;
        int const planeNumber = i->second;

        if (planeName != "") {
            metadata->add(maskPlanePrefix + planeName, planeNumber);
        }
    }
}

template <typename MaskPixelT>
typename Mask<MaskPixelT>::MaskPlaneDict Mask<MaskPixelT>::parseMaskPlaneMetadata(
        std::shared_ptr<dafBase::PropertySet const> metadata) {
    MaskPlaneDict newDict;

    // First, clear existing MaskPlane metadata
    typedef std::vector<std::string> NameList;
    NameList paramNames = metadata->paramNames(false);
    int numPlanesUsed = 0;  // number of planes used

    // Iterate over childless properties with names starting with maskPlanePrefix
    for (NameList::const_iterator i = paramNames.begin(); i != paramNames.end(); ++i) {
        if (i->compare(0, maskPlanePrefix.size(), maskPlanePrefix) == 0) {
            // split off maskPlanePrefix to obtain plane name
            std::string planeName = i->substr(maskPlanePrefix.size());
            int const planeId = metadata->getAsInt(*i);

            MaskPlaneDict::const_iterator plane = newDict.find(planeName);
            if (plane != newDict.end() && planeId != plane->second) {
                throw LSST_EXCEPT(pexExcept::RuntimeError, "File specifies plane " + planeName + " twice");
            }
            for (MaskPlaneDict::const_iterator j = newDict.begin(); j != newDict.end(); ++j) {
                if (planeId == j->second) {
                    throw LSST_EXCEPT(pexExcept::RuntimeError,
                                      str(boost::format("File specifies plane %s has same value (%d) as %s") %
                                          planeName % planeId % j->first));
                }
            }
            // build new entry
            if (numPlanesUsed >= getNumPlanesMax()) {
                // Max number of planes already allocated
                throw LSST_EXCEPT(
                        pexExcept::RuntimeError,
                        str(boost::format("Max number of planes (%1%) already used") % getNumPlanesMax()));
            }
            newDict[planeName] = planeId;
        }
    }
    return newDict;
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::printMaskPlanes() const {
    _maskDict->print();
}

/*
 * Static members of Mask
 */
template <typename MaskPixelT>
std::string const Mask<MaskPixelT>::maskPlanePrefix("MP_");

template <typename MaskPixelT>
std::shared_ptr<detail::MaskDict> Mask<MaskPixelT>::_maskPlaneDict() {
    return detail::MaskDict::getDefault();
}

//
// Explicit instantiations
//
template class Mask<MaskPixel>;
}  // namespace image
}  // namespace afw
}  // namespace lsst
