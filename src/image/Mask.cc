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

// Implementations of Mask class methods

/*
 * There are a number of classes defined here and in Mask.h
 *
 * The fundamental type visible to the user is Mask; a 2-d array of pixels.  Each of these pixels should
 * be thought of as a set of bits, and the names of these bits are given by MaskPlaneDict (which is
 * implemented as a std::map)
 *
 * Internally to this file, we have a MapWithHash which is like a std::map, but maintains a hash of its
 * contents;  this is used to check equality efficiently.
 *
 * We also have a MaskDict which isa MapWithHash, but also maintains a list of MaskDicts (or equivalently
 * MapWithHash) allowing us to iterate over these maps, updating them as needed.
 *
 * The list of MaskDicts is actually kept as a singleton of a helper class, DictState
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
#include "lsst/daf/base/Citizen.h"
#include "lsst/pex/exceptions.h"
#include "lsst/log/Log.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/Mask.h"

#include "lsst/afw/image/LsstImageTypes.h"

//
// for FITS code
//
#include "boost/mpl/vector.hpp"
#include "boost/gil/gil_all.hpp"
#include "lsst/afw/image/fits/fits_io.h"
#include "lsst/afw/image/fits/fits_io_mpl.h"

namespace afwGeom = lsst::afw::geom;
namespace dafBase = lsst::daf::base;
namespace pexExcept = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace image {
namespace detail {
class MaskDict;
}

namespace {
void setInitMaskBits(std::shared_ptr<detail::MaskDict> dict);
/*
 * A std::map that maintains a hash value of its contents
 *
 * We don't simply inherit from the std::map as we need to force the user to use add and remove;
 * we could inherit, make operator[] private, and never use MapWithHash via a base-class pointer
 * but it seemed simpler to only forward the functions we wish to support
 */
struct MapWithHash {
    typedef detail::MaskPlaneDict::value_type value_type;
    typedef detail::MaskPlaneDict::const_iterator const_iterator;

    MapWithHash(detail::MaskPlaneDict const& dict = detail::MaskPlaneDict())
            : _dict(dict), _hash(_calcHash()) {}
    ~MapWithHash() {}

    bool operator==(MapWithHash const& rhs) const { return _hash == rhs._hash; }

    const_iterator begin() const { return _dict.begin(); }
    const_iterator end() const { return _dict.end(); }
    const_iterator find(detail::MaskPlaneDict::key_type const& name) const { return _dict.find(name); }

    void add(std::string const& str, int val) {
        _dict[str] = val;
        _calcHash();
    }

    bool empty() const { return _dict.empty(); }

    void clear() { _dict.clear(); }

    std::size_t size() const { return _dict.size(); }

    void erase(std::string const& str) {
        if (_dict.find(str) != _dict.end()) {
            _dict.erase(str);
            _calcHash();
        }
    }

    detail::MaskPlaneDict const& getMaskPlaneDict() const { return _dict; }

    std::size_t getHash() const { return _hash; }

private:
    detail::MaskPlaneDict _dict;
    std::size_t _hash;

    // calculate the hash
    std::size_t _calcHash() {
        _hash = 0x0;
        for (const_iterator ptr = begin(); ptr != end(); ++ptr) {
            _hash = (_hash << 1) ^
                    boost::hash<std::string>()((*ptr).first + str(boost::format("%d") % ptr->second));
        }

        return _hash;
    }
};

bool operator!=(MapWithHash const& lhs, MapWithHash const& rhs) { return !(lhs == rhs); }

class DictState;  // forward declaration
}

namespace detail {
/*
 * A MaskDict is a MapWithHash, but additionally maintains a list of all live MaskDicts (the list is
 * actually kept in a singleton instance of DictState)
 */
class MaskDict : public MapWithHash {
    friend class ::lsst::afw::image::DictState;  // actually anonymous within lsst::afw::image; g++ is
                                                 // confused

    MaskDict() : MapWithHash() {}
    MaskDict(MapWithHash const* dict) : MapWithHash(*dict) {}

public:
    static std::shared_ptr<MaskDict> makeMaskDict();
    static std::shared_ptr<MaskDict> makeMaskDict(detail::MaskPlaneDict const& dict);
    static std::shared_ptr<MaskDict> setDefaultDict(std::shared_ptr<MaskDict> dict);

    std::shared_ptr<MaskDict> clone() const;

    ~MaskDict();

    int getUnusedPlane() const;
    int getMaskPlane(const std::string& name) const;

    void print() const {
        for (MapWithHash::const_iterator ptr = begin(); ptr != end(); ++ptr) {
            std::cout << "Plane " << ptr->second << " -> " << ptr->first << std::endl;
        }
    }

    static std::shared_ptr<MaskDict> incrDefaultVersion();
    static void listMaskDicts();
};
}

namespace {
/*
 * A struct to hold our global state, and for whose components
 * we can control the order of creation/destruction
 */
class DictState {
    friend class detail::MaskDict;

    typedef std::map<MapWithHash*, int> HandleList;

public:
    DictState() {
        _dictCounter = 0;
        _defaultMaskDict = std::shared_ptr<detail::MaskDict>(new detail::MaskDict);
        _dicts[_defaultMaskDict.get()] = _dictCounter++;
    }

    ~DictState() {
        _defaultMaskDict.reset();

        for (HandleList::iterator ptr = _dicts.begin(); ptr != _dicts.end(); ++ptr) {
            delete ptr->first;
        }
        _dicts.clear();
    }

    template <typename FunctorT>
    void forEachMaskDict(FunctorT func) {
        for (HandleList::const_iterator ptr = _dicts.begin(); ptr != _dicts.end(); ++ptr) {
            func(ptr->first);
        }
    }

private:
    std::shared_ptr<detail::MaskDict> getDefaultDict() {
        static bool first = true;

        if (first) {
            setInitMaskBits(_defaultMaskDict);

            first = false;
        }

        return _defaultMaskDict;
    }

    std::shared_ptr<detail::MaskDict> setDefaultDict(std::shared_ptr<detail::MaskDict> newDefaultMaskDict) {
        _defaultMaskDict = newDefaultMaskDict;

        return _defaultMaskDict;
    }

    void addDict(MapWithHash* dict) { _dicts[dict] = _dictCounter++; }

    void eraseDict(MapWithHash* dict) { _dicts.erase(dict); }

    std::shared_ptr<detail::MaskDict> incrDefaultVersion() {
        _defaultMaskDict = std::shared_ptr<detail::MaskDict>(new detail::MaskDict(*_defaultMaskDict.get()));
        addDict(_defaultMaskDict.get());

        return _defaultMaskDict;
    }

    std::shared_ptr<detail::MaskDict> _defaultMaskDict;  // default MaskDict to use
    HandleList _dicts;                                   // all the live MaskDicts
    int _dictCounter;
};

static DictState _state;
}

namespace detail {
/*
 * Implementation of MaskDict
 */
/*
 * Return the default dictionary, unless you provide mpd in which case you get one of
 * your very very own
 */
std::shared_ptr<MaskDict> MaskDict::makeMaskDict() { return _state.getDefaultDict(); }

std::shared_ptr<MaskDict> MaskDict::makeMaskDict(detail::MaskPlaneDict const& mpd) {
    std::shared_ptr<MaskDict> dict = _state.getDefaultDict();

    if (!mpd.empty()) {
        MapWithHash mwh(mpd);
        dict = std::shared_ptr<MaskDict>(new MaskDict(&mwh));
        _state.addDict(dict.get());
    }

    return dict;
}

std::shared_ptr<MaskDict> MaskDict::setDefaultDict(std::shared_ptr<MaskDict> dict) {
    return _state.setDefaultDict(dict);
}

std::shared_ptr<MaskDict> MaskDict::clone() const {
    std::shared_ptr<MaskDict> dict(new MaskDict(*this));

    _state.addDict(dict.get());

    return dict;
}

MaskDict::~MaskDict() { _state.eraseDict(this); }

std::shared_ptr<MaskDict> MaskDict::incrDefaultVersion() { return _state.incrDefaultVersion(); }

int MaskDict::getUnusedPlane() const {
    if (empty()) {
        return 0;
    }

    MapWithHash::const_iterator const it = std::max_element(
            begin(), end(),
            std::bind(std::less<int>(), std::bind(&MapWithHash::value_type::second, std::placeholders::_1),
                      std::bind(&MapWithHash::value_type::second, std::placeholders::_2)));
    assert(it != end());
    int id = it->second + 1;  // The maskPlane to use if there are no gaps

    for (int i = 0; i < id; ++i) {
        MapWithHash::const_iterator const it =  // is i already used in this Mask?
                std::find_if(
                        begin(), end(),
                        std::bind(std::equal_to<int>(),
                                  std::bind(&MapWithHash::value_type::second, std::placeholders::_1), i));
        if (it == end()) {  // Not used; so we'll use it
            return i;
        }
    }

    return id;
}

int detail::MaskDict::getMaskPlane(const std::string& name) const {
    MapWithHash::const_iterator i = find(name);

    return (i == end()) ? -1 : i->second;
}
}

namespace {
/*
 * Definition of the default mask bits
 *
 * N.b. this function is in an anonymous namespace, and is invisible to doxygen.  ALL mask
 * planes defined here should be documented with the Mask class in Mask.h
 */
void setInitMaskBits(std::shared_ptr<detail::MaskDict> dict) {
    int i = -1;
    dict->add("BAD", ++i);
    dict->add("SAT", ++i);       // should be SATURATED
    dict->add("INTRP", ++i);     // should be INTERPOLATED
    dict->add("CR", ++i);        //
    dict->add("EDGE", ++i);      //
    dict->add("DETECTED", ++i);  //
    dict->add("DETECTED_NEGATIVE", ++i);
    dict->add("SUSPECT", ++i);
    dict->add("NO_DATA", ++i);
}
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::_initializePlanes(MaskPlaneDict const& planeDefs) {
    LOGL_DEBUG("afw.image.Mask", "Number of mask planes: %d", getNumPlanesMax());

    _maskDict = detail::MaskDict::makeMaskDict(planeDefs);
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(unsigned int width, unsigned int height, MaskPlaneDict const& planeDefs)
        : ImageBase<MaskPixelT>(afwGeom::ExtentI(width, height)) {
    _initializePlanes(planeDefs);
    *this = 0x0;
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(unsigned int width, unsigned int height, MaskPixelT initialValue,
                       MaskPlaneDict const& planeDefs)
        : ImageBase<MaskPixelT>(afwGeom::ExtentI(width, height)) {
    _initializePlanes(planeDefs);
    *this = initialValue;
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(afwGeom::Extent2I const& dimensions, MaskPlaneDict const& planeDefs)
        : ImageBase<MaskPixelT>(dimensions) {
    _initializePlanes(planeDefs);
    *this = 0x0;
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(afwGeom::Extent2I const& dimensions, MaskPixelT initialValue,
                       MaskPlaneDict const& planeDefs)
        : ImageBase<MaskPixelT>(dimensions) {
    _initializePlanes(planeDefs);
    *this = initialValue;
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(afwGeom::Box2I const& bbox, MaskPlaneDict const& planeDefs)
        : ImageBase<MaskPixelT>(bbox) {
    _initializePlanes(planeDefs);
    *this = 0x0;
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(afwGeom::Box2I const& bbox, MaskPixelT initialValue, MaskPlaneDict const& planeDefs)
        : ImageBase<MaskPixelT>(bbox) {
    _initializePlanes(planeDefs);
    *this = initialValue;
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(Mask const& rhs, afwGeom::Box2I const& bbox, ImageOrigin const origin, bool const deep)
        : ImageBase<MaskPixelT>(rhs, bbox, origin, deep), _maskDict(rhs._maskDict) {}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(Mask const& rhs, bool deep)
        : ImageBase<MaskPixelT>(rhs, deep), _maskDict(rhs._maskDict) {}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(ndarray::Array<MaskPixelT, 2, 1> const& array, bool deep, geom::Point2I const& xy0)
        : image::ImageBase<MaskPixelT>(array, deep, xy0), _maskDict(detail::MaskDict::makeMaskDict()) {}

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

template <typename MaskPixelT>
Mask<MaskPixelT>& Mask<MaskPixelT>::operator=(MaskPixelT const rhs) {
    fill_pixels(_getRawView(), rhs);

    return *this;
}

#ifndef DOXYGEN  // doc for this section is already in header

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(std::string const& fileName, int hdu, std::shared_ptr<daf::base::PropertySet> metadata,
                       afw::geom::Box2I const& bbox, ImageOrigin origin, bool conformMasks)
        : ImageBase<MaskPixelT>(), _maskDict(detail::MaskDict::makeMaskDict()) {
    fits::Fits fitsfile(fileName, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    fitsfile.setHdu(hdu);
    *this = Mask(fitsfile, metadata, bbox, origin, conformMasks);
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(fits::MemFileManager& manager, int hdu,
                       std::shared_ptr<daf::base::PropertySet> metadata, afw::geom::Box2I const& bbox,
                       ImageOrigin origin, bool conformMasks)
        : ImageBase<MaskPixelT>(), _maskDict(detail::MaskDict::makeMaskDict()) {
    fits::Fits fitsfile(manager, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    fitsfile.setHdu(hdu);
    *this = Mask(fitsfile, metadata, bbox, origin, conformMasks);
}

template <typename MaskPixelT>
Mask<MaskPixelT>::Mask(fits::Fits& fitsfile, std::shared_ptr<daf::base::PropertySet> metadata,
                       afw::geom::Box2I const& bbox, ImageOrigin const origin, bool const conformMasks)
        : ImageBase<MaskPixelT>(), _maskDict(detail::MaskDict::makeMaskDict()) {
    // These are the permitted input file types
    typedef boost::mpl::vector<unsigned char, unsigned short, short, std::int32_t> fits_mask_types;

    if (!metadata) {
        metadata = std::shared_ptr<daf::base::PropertySet>(new daf::base::PropertyList);
    }

    fits_read_image<fits_mask_types>(fitsfile, *this, *metadata, bbox, origin);

    // look for mask planes in the file
    MaskPlaneDict fileMaskDict = parseMaskPlaneMetadata(metadata);
    std::shared_ptr<detail::MaskDict> fileMD = detail::MaskDict::makeMaskDict(fileMaskDict);

    if (*fileMD == *detail::MaskDict::makeMaskDict()) {  // file is already consistent with Mask
        return;
    }

    if (conformMasks) {  // adopt the definitions in the file
        _maskDict = detail::MaskDict::setDefaultDict(fileMD);
    }

    conformMaskPlanes(fileMaskDict);  // convert planes defined by fileMaskDict to the order
                                      // defined by Mask::_maskPlaneDict
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
                                 std::shared_ptr<lsst::daf::base::PropertySet const> metadata_i) const {
    std::shared_ptr<dafBase::PropertySet> metadata;
    if (metadata_i) {
        metadata = metadata_i->deepCopy();
    } else {
        metadata = std::shared_ptr<dafBase::PropertySet>(new dafBase::PropertyList());
    }
    addMaskPlanesToMetadata(metadata);
    //
    // Add WCS with (X0, Y0) information
    //
    std::shared_ptr<dafBase::PropertySet> wcsAMetadata =
            detail::createTrivialWcsAsPropertySet(detail::wcsNameForXY0, this->getX0(), this->getY0());
    metadata->combine(wcsAMetadata);

    fits_write_image(fitsfile, *this, metadata);
}

#endif  // !DOXYGEN

namespace {
struct addPlaneFunctor {
    addPlaneFunctor(std::string const& name, int id) : _name(name), _id(id) {}

    void operator()(MapWithHash* dict) {
        detail::MaskPlaneDict::const_iterator const it =  // is id already used in this Mask?
                std::find_if(
                        dict->begin(), dict->end(),
                        std::bind(std::equal_to<int>(), std::bind(&detail::MaskPlaneDict::value_type::second,
                                                                  std::placeholders::_1),
                                  _id));
        if (it != dict->end()) {  // mask plane is already in use
            return;
        }

        if (dict->find(_name) == dict->end()) {  // not already set
            dict->add(_name, _id);
        }
    }

    std::string const& _name;
    int _id;
};
}

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

    _state.forEachMaskDict(addPlaneFunctor(name, id));

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
    if (detail::MaskDict::makeMaskDict()->getMaskPlane(name) < 0) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError,
                          str(boost::format("Plane %s doesn't exist in the default Mask") % name));
    }

    detail::MaskDict::incrDefaultVersion();  // leave current Masks alone
    _maskPlaneDict()->erase(name);
}

template <typename MaskPixelT>
void Mask<MaskPixelT>::removeAndClearMaskPlane(const std::string& name, bool const removeFromDefault

                                               ) {
    clearMaskPlane(getMaskPlane(name));  // clear this bits in this Mask

    if (_maskDict == detail::MaskDict::makeMaskDict() && removeFromDefault) {  // we are the default
        ;
    } else {
        _maskDict = _maskDict->clone();
    }

    _maskDict->erase(name);

    if (removeFromDefault && detail::MaskDict::makeMaskDict()->getMaskPlane(name) >= 0) {
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
    std::shared_ptr<detail::MaskDict> currentMD = detail::MaskDict::makeMaskDict(currentPlaneDict);

    if (*_maskDict == *currentMD) {
        if (*detail::MaskDict::makeMaskDict() == *_maskDict) {
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
    _maskDict = detail::MaskDict::makeMaskDict();
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
    return detail::MaskDict::makeMaskDict();
}

//
// Explicit instantiations
//
template class Mask<MaskPixel>;
}
}
}
