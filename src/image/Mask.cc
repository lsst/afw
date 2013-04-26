// -*- lsst-c++ -*-

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
 
/// \file
/// \brief Implementations of Mask class methods

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
#include "boost/lambda/lambda.hpp"
#pragma clang diagnostic pop
#include "boost/format.hpp"
#include "boost/filesystem/path.hpp"

#include "boost/functional/hash.hpp"

#include "boost/bind.hpp"

#include "lsst/daf/base.h"
#include "lsst/daf/base/Citizen.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/Mask.h"

#include "lsst/afw/image/LsstImageTypes.h"

/************************************************************************************************************/
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
namespace pexLog = lsst::pex::logging;

/************************************************************************************************************/

namespace lsst { namespace afw { namespace image {
    namespace detail {
        class MaskDict;
    }

namespace {
    void setInitMaskBits(PTR(detail::MaskDict) dict);
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

        MapWithHash(detail::MaskPlaneDict const& dict=detail::MaskPlaneDict()) :
            _dict(dict), _hash(_calcHash()) {
        }
        ~MapWithHash() { }

        bool operator==(MapWithHash const& rhs) const {
            return _hash == rhs._hash;
        }

        const_iterator begin() const { return _dict.begin(); }
        const_iterator end() const { return _dict.end(); }
        const_iterator find(detail::MaskPlaneDict::key_type const& name) const { return _dict.find(name); }

        void add(std::string const& str, int val) {
            _dict[str] = val;
            _calcHash();
        }

        bool empty() const {
            return _dict.empty();
        }
        
        void clear() {
            _dict.clear();
        }
        
        std::size_t size() const {
            return _dict.size();
        }

        void erase(std::string const& str) {
            if (_dict.find(str) != _dict.end()) {
                _dict.erase(str);
                _calcHash();
            }
        }

        detail::MaskPlaneDict const& getMaskPlaneDict() const {
            return _dict;
        }
        
        std::size_t getHash() const {
            return _hash;
        }
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

    bool operator!=(MapWithHash const& lhs, MapWithHash const& rhs) {
        return !(lhs == rhs);
    }

    class DictState;                   // forward declaration
}

namespace detail {
/*
 * A MaskDict is a MapWithHash, but additionally maintains a list of all live MaskDicts (the list is
 * actually kept in a singleton instance of DictState)
 */
class MaskDict : public MapWithHash {
    friend class ::lsst::afw::image::DictState; // actually anonymous within lsst::afw::image; g++ is confused

    MaskDict() : MapWithHash() {}
    MaskDict(MapWithHash const* dict) : MapWithHash(*dict) {}
public:
    static PTR(MaskDict) makeMaskDict();
    static PTR(MaskDict) makeMaskDict(detail::MaskPlaneDict const &dict);
    static PTR(MaskDict) setDefaultDict(PTR(MaskDict) dict);

    PTR(MaskDict) clone() const;

    ~MaskDict();

    int getUnusedPlane() const;
    int getMaskPlane(const std::string& name) const;

    void print() const {
        for (MapWithHash::const_iterator ptr = begin(); ptr != end(); ++ptr) {
            std::cout << "Plane " << ptr->second << " -> " << ptr->first << std::endl;
        }
    }

    static PTR(MaskDict) incrDefaultVersion();
    static void listMaskDicts();
};
}

/************************************************************************************************************/

namespace {
    /*
     * A struct to hold our global state, and for whose components
     * we can control the order of creation/destruction
     */
    class DictState {
        friend class detail::MaskDict;

        typedef std::map<MapWithHash *, int> HandleList;

    public:
        DictState() {
            _dictCounter = 0;
            _defaultMaskDict = PTR(detail::MaskDict)(new detail::MaskDict);
            _dicts[_defaultMaskDict.get()] = _dictCounter++;
        }

        ~DictState() {
            _defaultMaskDict.reset();

            for (HandleList::iterator ptr = _dicts.begin(); ptr != _dicts.end(); ++ptr) {
                delete ptr->first;
            }
            _dicts.clear();
        }

        template<typename FunctorT>
        void forEachMaskDict(FunctorT func) {
            for (HandleList::const_iterator ptr = _dicts.begin(); ptr != _dicts.end(); ++ptr) {
                func(ptr->first);
            }
        }

    private:
        PTR(detail::MaskDict) getDefaultDict() {
            static bool first = true;

            if (first) {
                setInitMaskBits(_defaultMaskDict);
                
                first = false;
            }

            return _defaultMaskDict;
        }

        PTR(detail::MaskDict) setDefaultDict(PTR(detail::MaskDict) newDefaultMaskDict) {
            _defaultMaskDict = newDefaultMaskDict;

            return _defaultMaskDict;
        }

        void addDict(MapWithHash *dict) {
            _dicts[dict] = _dictCounter++;
        }
            
        void eraseDict(MapWithHash *dict) {
            _dicts.erase(dict);
        }
        
        PTR(detail::MaskDict) incrDefaultVersion() {
            _defaultMaskDict = PTR(detail::MaskDict)(new detail::MaskDict(*_defaultMaskDict.get()));
            addDict(_defaultMaskDict.get());

            return _defaultMaskDict;
        }

        PTR(detail::MaskDict) _defaultMaskDict; // default MaskDict to use
        HandleList _dicts;                            // all the live MaskDicts
        int _dictCounter;
    };

    static DictState _state;
}

/************************************************************************************************************/
namespace detail {
/*
 * Implementation of MaskDict
 */
/*
 * Return the default dictionary, unless you provide mpd in which case you get one of
 * your very very own
 */
PTR(MaskDict)
MaskDict::makeMaskDict()
{
    return _state.getDefaultDict();
}

PTR(MaskDict)
MaskDict::makeMaskDict(detail::MaskPlaneDict const& mpd)
{
    PTR(MaskDict) dict = _state.getDefaultDict();

    if (!mpd.empty()) {
        MapWithHash mwh(mpd);
        dict = PTR(MaskDict)(new MaskDict(&mwh));
        _state.addDict(dict.get());
    }

    return dict;
}

PTR(MaskDict)
MaskDict::setDefaultDict(PTR(MaskDict) dict)
{
    return _state.setDefaultDict(dict);
}
            
PTR(MaskDict) MaskDict::clone() const {
    PTR(MaskDict) dict(new MaskDict(*this));
    
    _state.addDict(dict.get());

    return dict;
}

MaskDict::~MaskDict() {
    _state.eraseDict(this);
}

PTR(MaskDict) MaskDict::incrDefaultVersion() {
    return _state.incrDefaultVersion();
}

int
MaskDict::getUnusedPlane() const
{
    if (empty()) {
        return 0;
    }

    MapWithHash::const_iterator const it =
        std::max_element(begin(), end(), boost::bind(std::less<int>(),
                                                   boost::bind(&MapWithHash::value_type::second, _1),
                                                   boost::bind(&MapWithHash::value_type::second, _2)
                                                  )
                        );
    assert(it != end());
    int id = it->second + 1;        // The maskPlane to use if there are no gaps
        
    for (int i = 0; i < id; ++i) {
        MapWithHash::const_iterator const it = // is i already used in this Mask?
            std::find_if(begin(), end(), boost::bind(std::equal_to<int>(),
                                                   boost::bind(&MapWithHash::value_type::second, _1), i));
        if (it == end()) {              // Not used; so we'll use it
            return i;
        }
    }

    return id;
}

int
detail::MaskDict::getMaskPlane(const std::string& name) const
{
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
    void
    setInitMaskBits(PTR(detail::MaskDict) dict)
    {
        int i = -1;
        dict->add("BAD", ++i);
        dict->add("SAT", ++i);           // should be SATURATED
        dict->add("INTRP", ++i);         // should be INTERPOLATED
        dict->add("CR", ++i);            // 
        dict->add("EDGE", ++i);          // 
        dict->add("DETECTED", ++i);      // 
        dict->add("DETECTED_NEGATIVE", ++i);
        dict->add("SUSPECT", ++i);
    }
}

/**
 * \brief Initialise mask planes; called by constructors
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::_initializePlanes(MaskPlaneDict const& planeDefs) {
    pexLog::Trace("afw.Mask", 5, boost::format("Number of mask planes: %d") % getNumPlanesMax());

    _maskDict = detail::MaskDict::makeMaskDict(planeDefs);
}

/**
 * \brief Construct a Mask initialized to 0x0
 */
template<typename MaskPixelT>
Mask<MaskPixelT>::Mask(
    unsigned int width,                 ///< number of columns
    unsigned int height,                ///< number of rows
    MaskPlaneDict const& planeDefs ///< desired mask planes
) :
    ImageBase<MaskPixelT>(afwGeom::ExtentI(width, height)) {
    _initializePlanes(planeDefs);
    *this = 0x0;
}

/**
 * \brief Construct a Mask initialized to a specified value
 */
template<typename MaskPixelT>
Mask<MaskPixelT>::Mask(
    unsigned int width,                 ///< number of columns
    unsigned int height,                ///< number of rows
    MaskPixelT initialValue,            ///< Initial value
    MaskPlaneDict const& planeDefs      ///< desired mask planes
) :
    ImageBase<MaskPixelT>(afwGeom::ExtentI(width, height)) {
    _initializePlanes(planeDefs);
    *this = initialValue;
}

/**
 * \brief Construct a Mask initialized to 0x0
 */
template<typename MaskPixelT>
Mask<MaskPixelT>::Mask(
    afwGeom::Extent2I const & dimensions, ///< Number of columns, rows
    MaskPlaneDict const& planeDefs ///< desired mask planes
) :
    ImageBase<MaskPixelT>(dimensions) {
    _initializePlanes(planeDefs);
    *this = 0x0;
}

/**
 * \brief Construct a Mask initialized to a specified value
 */
template<typename MaskPixelT>
Mask<MaskPixelT>::Mask(
    afwGeom::Extent2I const & dimensions, ///< Number of columns, rows
    MaskPixelT initialValue, ///< Initial value
    MaskPlaneDict const& planeDefs ///< desired mask planes
) :
    ImageBase<MaskPixelT>(dimensions) {
    _initializePlanes(planeDefs);
    *this = initialValue;
}

/**
 * \brief Construct a Mask initialized to 0x0
 */
template<typename MaskPixelT>
Mask<MaskPixelT>::Mask(
    afwGeom::Box2I const & bbox, ///< Desired number of columns/rows and origin
    MaskPlaneDict const& planeDefs ///< desired mask planes
) :
    ImageBase<MaskPixelT>(bbox) {
    _initializePlanes(planeDefs);
    *this = 0x0;
}

/**
 * \brief Construct a Mask initialized to a specified value
 */
template<typename MaskPixelT>
Mask<MaskPixelT>::Mask(
    afwGeom::Box2I const & bbox, ///< Desired number of columns/rows and origin
    MaskPixelT initialValue, ///< Initial value
    MaskPlaneDict const& planeDefs ///< desired mask planes
) :
    ImageBase<MaskPixelT>(bbox) {
    _initializePlanes(planeDefs);
    *this = initialValue;
}

/**
 * \brief Construct a Mask from a subregion of another Mask
 */
template<typename MaskPixelT>
Mask<MaskPixelT>::Mask(
    Mask const &rhs,    ///< mask to copy
    afwGeom::Box2I const &bbox,   ///< subregion to copy
    ImageOrigin const origin,  ///< coordinate system of the bbox
    bool const deep     ///< deep copy? (construct a view with shared pixels if false)
) :
    ImageBase<MaskPixelT>(rhs, bbox, origin, deep), _maskDict(rhs._maskDict) {
}

/**
 * \brief Construct a Mask from another Mask
 */
template<typename MaskPixelT>
Mask<MaskPixelT>::Mask(
    Mask const& rhs,    ///< mask to copy
    bool deep           ///< deep copy? (construct a view with shared pixels if false)
) :
    ImageBase<MaskPixelT>(rhs, deep), _maskDict(rhs._maskDict) {
}

template<typename MaskPixelT>
Mask<MaskPixelT>::Mask(ndarray::Array<MaskPixelT,2,1> const & array, bool deep,
                       geom::Point2I const & xy0) :
        image::ImageBase<MaskPixelT>(array, deep, xy0),
        _maskDict(detail::MaskDict::makeMaskDict()) {
}

/************************************************************************************************************/

template<typename PixelT>
void Mask<PixelT>::swap(Mask &rhs) {
    using std::swap;                    // See Meyers, Effective C++, Item 25

    ImageBase<PixelT>::swap(rhs);
    swap(_maskDict, rhs._maskDict);
}

template<typename PixelT>
void swap(Mask<PixelT>& a, Mask<PixelT>& b) {
    a.swap(b);
}

template<typename MaskPixelT>
Mask<MaskPixelT>& Mask<MaskPixelT>::operator=(const Mask<MaskPixelT>& rhs) {
    Mask tmp(rhs);
    swap(tmp);                        // See Meyers, Effective C++, Item 11
    
    return *this;
}

template<typename MaskPixelT>
Mask<MaskPixelT>& Mask<MaskPixelT>::operator=(MaskPixelT const rhs) {
    fill_pixels(_getRawView(), rhs);

    return *this;
}

#ifndef DOXYGEN // doc for this section is already in header

template<typename MaskPixelT>
Mask<MaskPixelT>::Mask(
    std::string const & fileName,
    int hdu,
    PTR(daf::base::PropertySet) metadata,
    afw::geom::Box2I const & bbox,
    ImageOrigin origin,
    bool conformMasks
) : ImageBase<MaskPixelT>(), _maskDict(detail::MaskDict::makeMaskDict()) {
    fits::Fits fitsfile(fileName, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    fitsfile.setHdu(hdu);
    *this = Mask(fitsfile, metadata, bbox, origin, conformMasks);
}

template<typename MaskPixelT>
Mask<MaskPixelT>::Mask(
    fits::MemFileManager & manager,
    int hdu,
    PTR(daf::base::PropertySet) metadata,
    afw::geom::Box2I const & bbox,
    ImageOrigin origin,
    bool conformMasks
) : ImageBase<MaskPixelT>(), _maskDict(detail::MaskDict::makeMaskDict()) {

    fits::Fits fitsfile(manager, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    fitsfile.setHdu(hdu);
    *this = Mask(fitsfile, metadata, bbox, origin, conformMasks);
}

template<typename MaskPixelT>
Mask<MaskPixelT>::Mask(
    fits::Fits & fitsfile,
    PTR(daf::base::PropertySet) metadata,
    afw::geom::Box2I const & bbox,
    ImageOrigin const origin,
    bool const conformMasks
) :
    ImageBase<MaskPixelT>(), _maskDict(detail::MaskDict::makeMaskDict()) 
{
    // These are the permitted input file types
    typedef boost::mpl::vector<
        unsigned char, 
        unsigned short,
        short
    >fits_mask_types;

    if (!metadata) {
        metadata = PTR(daf::base::PropertySet)(new daf::base::PropertyList);
    }

    fits_read_image<fits_mask_types>(fitsfile, *this, *metadata, bbox, origin);

    // look for mask planes in the file
    MaskPlaneDict fileMaskDict = parseMaskPlaneMetadata(metadata); 
    PTR(detail::MaskDict) fileMD = detail::MaskDict::makeMaskDict(fileMaskDict);

    if (*fileMD == *detail::MaskDict::makeMaskDict()) { // file is already consistent with Mask
        return;
    }
    
    if (conformMasks) {                 // adopt the definitions in the file
        _maskDict = detail::MaskDict::setDefaultDict(fileMD);
    }

    conformMaskPlanes(fileMaskDict);    // convert planes defined by fileMaskDict to the order
                                        // defined by Mask::_maskPlaneDict
}

template<typename MaskPixelT>
void Mask<MaskPixelT>::writeFits(
    std::string const & fileName,
    CONST_PTR(lsst::daf::base::PropertySet) metadata_i,
    std::string const & mode
) const {
    fits::Fits fitsfile(fileName, mode, fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, metadata_i);
}

template<typename MaskPixelT>
void Mask<MaskPixelT>::writeFits(
    fits::MemFileManager & manager,
    CONST_PTR(lsst::daf::base::PropertySet) metadata_i,
    std::string const & mode
) const {
    fits::Fits fitsfile(manager, mode, fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, metadata_i);
}

template<typename MaskPixelT>
void Mask<MaskPixelT>::writeFits(
    fits::Fits & fitsfile,
    CONST_PTR(lsst::daf::base::PropertySet) metadata_i
) const {

    PTR(dafBase::PropertySet) metadata;
    if (metadata_i) {
        metadata = metadata_i->deepCopy();
    } else {
        metadata = PTR(dafBase::PropertySet)(new dafBase::PropertyList());
    }
    addMaskPlanesToMetadata(metadata);
    //
    // Add WCS with (X0, Y0) information
    //
    PTR(dafBase::PropertySet) wcsAMetadata = detail::createTrivialWcsAsPropertySet(
        detail::wcsNameForXY0, this->getX0(), this->getY0()
    );
    metadata->combine(wcsAMetadata);

    fits_write_image(fitsfile, *this, metadata);
}

#endif // !DOXYGEN

namespace {
    struct addPlaneFunctor {
        addPlaneFunctor(std::string const& name, int id) : _name(name), _id(id) {}
        
        void operator()(MapWithHash *dict) {
            detail::MaskPlaneDict::const_iterator const it = // is id already used in this Mask?
                std::find_if(dict->begin(), dict->end(),
                             boost::bind(std::equal_to<int>(),
                                       boost::bind(&detail::MaskPlaneDict::value_type::second, _1), _id));
            if (it != dict->end()) {          // mask plane is already in use
                return;
            }
            
            if (dict->find(_name) == dict->end()) { // not already set
                dict->add(_name, _id);
            }
        }

        std::string const& _name;
        int _id;
    };
}

template<typename MaskPixelT>
int Mask<MaskPixelT>::addMaskPlane(const std::string& name)
{
    int id = getMaskPlaneNoThrow(name); // see if the plane is already available

    if (id < 0) {                       // doesn't exist
        id = _maskPlaneDict()->getUnusedPlane();
    }

    // build new entry, adding the plane to all Masks where this is no contradiction

    if (id >= getNumPlanesMax()) {      // Max number of planes is already allocated
        throw LSST_EXCEPT(pexExcept::RuntimeErrorException,
                          str(boost::format("Max number of planes (%1%) already used") % getNumPlanesMax()));
    }

    _state.forEachMaskDict(addPlaneFunctor(name, id));    

    return id;
}

/**
 * \brief set the name of a mask plane, with minimal checking.
 *
 * This is a private function and is mainly used by setMaskPlaneMetadata
 */
template<typename MaskPixelT>
int Mask<MaskPixelT>::addMaskPlane(
    std::string name,   ///< new name of mask plane
    int planeId         ///< ID of mask plane to be (re)named
) {
    if (planeId < 0 || planeId >= getNumPlanesMax()) {
        throw LSST_EXCEPT(pexExcept::RangeErrorException,
                     str(boost::format("mask plane ID must be between 0 and %1%") % (getNumPlanesMax() - 1)));
    }

    _maskPlaneDict()->add(name, planeId);

    return planeId;
}

/**
 * Return the Mask's maskPlaneDict
 */
template<typename MaskPixelT>
detail::MaskPlaneDict const&
Mask<MaskPixelT>::getMaskPlaneDict() const
{
    return _maskDict->getMaskPlaneDict();
}

template<typename MaskPixelT>
void Mask<MaskPixelT>::removeMaskPlane(const std::string& name)
{
    if (detail::MaskDict::makeMaskDict()->getMaskPlane(name) < 0) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,
                          str(boost::format("Plane %s doesn't exist in the default Mask") % name));
    }

    detail::MaskDict::incrDefaultVersion(); // leave current Masks alone
    _maskPlaneDict()->erase(name);
}

/**
 * \brief Clear all pixels of the specified mask and remove the plane from the mask plane dictionary;
 * optionally remove the plane from the default dictionary too
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if plane is invalid
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::removeAndClearMaskPlane(const std::string& name, ///< name of maskplane
                                                         bool const removeFromDefault ///< remove from default
                                        ///< mask plane dictionary too
                                                        )
{
    clearMaskPlane(getMaskPlane(name)); // clear this bits in this Mask

    if (_maskDict == detail::MaskDict::makeMaskDict() && removeFromDefault) { // we are the default
        ;
    } else {
        _maskDict = _maskDict->clone();
    }

    _maskDict->erase(name);

    if (removeFromDefault && detail::MaskDict::makeMaskDict()->getMaskPlane(name) >= 0) {
        removeMaskPlane(name);
    }
}

/**
 * \brief Return the bitmask corresponding to a plane ID, or 0 if invalid
 */
template<typename MaskPixelT>
MaskPixelT Mask<MaskPixelT>::getBitMaskNoThrow(int planeId) {
    return (planeId >= 0 && planeId < getNumPlanesMax()) ? (1 << planeId) : 0;
}

/**
 * \brief Return the bitmask corresponding to plane ID
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if plane is invalid
 */
template<typename MaskPixelT>
MaskPixelT Mask<MaskPixelT>::getBitMask(int planeId) {
    MaskPlaneDict const& mpd = _maskPlaneDict()->getMaskPlaneDict();
    
    for (MaskPlaneDict::const_iterator i = mpd.begin(); i != mpd.end(); ++i) {
        if (planeId == i->second) {
            MaskPixelT const bitmask = getBitMaskNoThrow(planeId);
            if (bitmask == 0) {         // failed
                break;
            }
            return bitmask;
        }
    }
    throw LSST_EXCEPT(pexExcept::InvalidParameterException,
                      str(boost::format("Invalid mask plane ID: %d") % planeId));
}

/**
 * \brief Return the mask plane number corresponding to a plane name
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if plane is invalid
 */
template<typename MaskPixelT>
int Mask<MaskPixelT>::getMaskPlane(const std::string& name) {
    int const plane = getMaskPlaneNoThrow(name);
    
    if (plane < 0) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,
                          str(boost::format("Invalid mask plane name: %s") % name));
    } else {
        return plane;
    }
}

/**
 * \brief Return the mask plane number corresponding to a plane name, or -1 if not found
 */
template<typename MaskPixelT>
int Mask<MaskPixelT>::getMaskPlaneNoThrow(const std::string& name) {
    return _maskPlaneDict()->getMaskPlane(name);
}

/**
 * \brief Return the bitmask corresponding to a plane name
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if plane is invalid
 */
template<typename MaskPixelT>
MaskPixelT Mask<MaskPixelT>::getPlaneBitMask(const std::string& name) {
    return getBitMask(getMaskPlane(name));
}

/**
 * \brief Return the bitmask corresponding to a vector of plane names OR'd together
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if plane is invalid
 */
template<typename MaskPixelT>
MaskPixelT Mask<MaskPixelT>::getPlaneBitMask(const std::vector<std::string> &name) {
    MaskPixelT mpix = 0x0;
    for (std::vector<std::string>::const_iterator it = name.begin(); it != name.end(); ++it) {
        mpix |= getBitMask(getMaskPlane(*it));
    }
    return mpix;
}

/**
 * \brief Reset the maskPlane dictionary
 */
template<typename MaskPixelT>
int Mask<MaskPixelT>::getNumPlanesUsed()
{
    return _maskPlaneDict()->size();
}

/**
 * \brief Reset the maskPlane dictionary
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::clearMaskPlaneDict() {
    _maskPlaneDict()->clear();
}

/**
 * \brief Clear all the pixels
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::clearAllMaskPlanes() {
    *this = 0;
}

/**
 * \brief Clear the specified bit in all pixels
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::clearMaskPlane(int planeId) {
    *this &= ~getBitMask(planeId);
}

/**
 * \brief Adjust this mask to conform to the standard Mask class's mask plane dictionary,
 * adding any new mask planes to the standard.
 *
 * Ensures that this mask (presumably from some external source) has the same plane assignments
 * as the Mask class. If a change in plane assignments is needed, the bits within each pixel
 * are permuted as required.  The provided currentPlaneDict describes the true state of the bits
 * in this Mask's pixels and overrides its current MaskDict
 *
 * Any new mask planes found in this mask are added to unused slots in the Mask class's mask plane dictionary.
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::conformMaskPlanes(
    MaskPlaneDict const &currentPlaneDict   ///< mask plane dictionary currently in use for this mask
                                                  )
{
    PTR(detail::MaskDict) currentMD = detail::MaskDict::makeMaskDict(currentPlaneDict);

    if (*_maskDict == *currentMD) {
        if (*detail::MaskDict::makeMaskDict() == *_maskDict) {
            return;   // nothing to do
        }
    } else {
        //
        // Find out which planes need to be permuted
        //
        MaskPixelT keepBitmask = 0;       // mask of bits to keep
        MaskPixelT canonicalMask[sizeof(MaskPixelT)*8]; // bits in lsst::afw::image::Mask that should be
        MaskPixelT currentMask[sizeof(MaskPixelT)*8];   //           mapped to these bits
        int numReMap = 0;

        for (MaskPlaneDict::const_iterator i = currentPlaneDict.begin(); i != currentPlaneDict.end() ; i++) {
            std::string const name = i->first; // name of mask plane
            int const currentPlaneNumber = i->second; // plane number currently in use
            int canonicalPlaneNumber = getMaskPlaneNoThrow(name); // plane number in lsst::afw::image::Mask

            if (canonicalPlaneNumber < 0) {                  // no such plane; add it
                canonicalPlaneNumber = addMaskPlane(name);
            }
        
            if (canonicalPlaneNumber == currentPlaneNumber) {
                keepBitmask |= getBitMask(canonicalPlaneNumber); // bit is unchanged, so preserve it
            } else {
                canonicalMask[numReMap] = getBitMask(canonicalPlaneNumber);
                currentMask[numReMap]   = getBitMaskNoThrow(currentPlaneNumber);
                numReMap++;
            }
        }

        // Now loop over all pixels in Mask
        if (numReMap > 0) {
            for (int r = 0; r != this->getHeight(); ++r) { // "this->": Meyers, Effective C++, Item 43
                for (typename Mask::x_iterator ptr = this->row_begin(r), end = this->row_end(r);
                     ptr != end; ++ptr) {
                    MaskPixelT const pixel = *ptr;

                    MaskPixelT newPixel = pixel & keepBitmask; // value of invariant mask bits
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

/************************************************************************************************************/

/**
 * \brief get a reference to the specified pixel
 */
template<typename MaskPixelT>
typename ImageBase<MaskPixelT>::PixelReference Mask<MaskPixelT>::operator()(
    int x,  ///< x index
    int y   ///< y index
) {
    return this->ImageBase<MaskPixelT>::operator()(x, y);
}

/**
 * \brief get a reference to the specified pixel checking array bounds
 */
template<typename MaskPixelT>
typename ImageBase<MaskPixelT>::PixelReference Mask<MaskPixelT>::operator()(
    int x,                              ///< x index
    int y,                              ///< y index
    CheckIndices const& check ///< Check array bounds?
) {
    return this->ImageBase<MaskPixelT>::operator()(x, y, check);
}

/**
 * \brief get the specified pixel (const version)
 */
template<typename MaskPixelT>
typename ImageBase<MaskPixelT>::PixelConstReference Mask<MaskPixelT>::operator()(
    int x,  ///< x index
    int y   ///< y index
) const {
    return this->ImageBase<MaskPixelT>::operator()(x, y);
}

/**
 * \brief get the specified pixel with array checking (const version)
 */
template<typename MaskPixelT>
typename ImageBase<MaskPixelT>::PixelConstReference Mask<MaskPixelT>::operator()(
    int x,                              ///< x index
    int y,                              ///< y index
    CheckIndices const& check ///< Check array bounds?
) const {
    return this->ImageBase<MaskPixelT>::operator()(x, y, check);
}

/**
 * \brief is the specified mask plane set in the specified pixel?
 */
template<typename MaskPixelT>
bool Mask<MaskPixelT>::operator()(
    int x,      ///< x index
    int y,      ///< y index
    int planeId ///< plane ID
) const {
    // !! converts an int to a bool
    return !!(this->ImageBase<MaskPixelT>::operator()(x, y) & getBitMask(planeId));
}

/**
 * \brief is the specified mask plane set in the specified pixel, checking array bounds?
 */
template<typename MaskPixelT>
bool Mask<MaskPixelT>::operator()(
    int x,                              ///< x index
    int y,                              ///< y index
    int planeId,                        ///< plane ID
    CheckIndices const& check ///< Check array bounds?
) const {
    // !! converts an int to a bool
    return !!(this->ImageBase<MaskPixelT>::operator()(x, y, check) & getBitMask(planeId));
}

/************************************************************************************************************/
//
// Check that masks have the same dictionary version
//
// @throw lsst::pex::exceptions::RuntimeErrorException
//
template<typename MaskPixelT>
void Mask<MaskPixelT>::checkMaskDictionaries(Mask<MaskPixelT> const &other) {
    if (*_maskDict != *other._maskDict) {
        throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "Mask dictionaries do not match");
    }
}        

/************************************************************************************************************/
//
// N.b. We could use the STL, but I find boost::lambda clearer, and more easily extended
// to e.g. setting random numbers
//    transform_pixels(_getRawView(), _getRawView(), lambda::ret<PixelT>(lambda::_1 + val));
// is equivalent to
//    transform_pixels(_getRawView(), _getRawView(), std::bind2nd(std::plus<PixelT>(), val));
//
namespace bl = boost::lambda;

/**
 * \brief OR a bitmask into a Mask
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::operator|=(MaskPixelT const val) {
    transform_pixels(_getRawView(), _getRawView(), bl::ret<MaskPixelT>(bl::_1 | val));
}

/**
 * \brief OR a Mask into a Mask
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::operator|=(Mask const &rhs) {
    checkMaskDictionaries(rhs);

    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(pexExcept::LengthErrorException,
                          str(boost::format("Images are of different size, %dx%d v %dx%d") %
                              this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()));
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), bl::ret<MaskPixelT>(bl::_1 | bl::_2));
}

/**
 * \brief AND a bitmask into a Mask
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::operator&=(MaskPixelT const val) {
    transform_pixels(_getRawView(), _getRawView(), bl::ret<MaskPixelT>(bl::_1 & val));
}

/**
 * \brief AND a Mask into a Mask
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::operator&=(Mask const &rhs) {
    checkMaskDictionaries(rhs);

    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(pexExcept::LengthErrorException,
                          str(boost::format("Images are of different size, %dx%d v %dx%d") %
                              this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()));
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), bl::ret<MaskPixelT>(bl::_1 & bl::_2));
}

/**
 * \brief XOR a bitmask into a Mask
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::operator^=(MaskPixelT const val) {
    transform_pixels(_getRawView(), _getRawView(), bl::ret<MaskPixelT>(bl::_1 ^ val));
}

/**
 * \brief XOR a Mask into a Mask
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::operator^=(Mask const &rhs) {
    checkMaskDictionaries(rhs);

    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(pexExcept::LengthErrorException,
                          str(boost::format("Images are of different size, %dx%d v %dx%d") %
                              this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()));
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), bl::ret<MaskPixelT>(bl::_1 ^ bl::_2));
}

/**
 * \brief Set the bit specified by "planeId" for pixels (x0, y) ... (x1, y)
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::setMaskPlaneValues(int const planeId,
                                                    int const x0, int const x1, int const y) {
    MaskPixelT const bitMask = getBitMask(planeId);
    
    for (int x = x0; x <= x1; x++) {
        operator()(x, y) = operator()(x, y) | bitMask;
    }
}

/**
 * \brief Given a PropertySet, replace any existing MaskPlane assignments with the current ones.
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::addMaskPlanesToMetadata(PTR(dafBase::PropertySet) metadata) {
    if (!metadata) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "Null PTR(PropertySet)");
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
    for (MaskPlaneDict::const_iterator i = mpd.begin(); i != mpd.end() ; ++i) {
        std::string const& planeName = i->first;
        int const planeNumber = i->second;

        if (planeName != "") {
            metadata->add(maskPlanePrefix + planeName, planeNumber);
        }
    }
}

/**
 * \brief Given a PropertySet that contains the MaskPlane assignments, setup the MaskPlanes.
 *
 * @returns a dictionary of mask plane name: plane ID
 */
template<typename MaskPixelT>
typename Mask<MaskPixelT>::MaskPlaneDict Mask<MaskPixelT>::parseMaskPlaneMetadata(
    CONST_PTR(dafBase::PropertySet) metadata ///< metadata from a Mask
) {
    MaskPlaneDict newDict;

    // First, clear existing MaskPlane metadata
    typedef std::vector<std::string> NameList;
    NameList paramNames = metadata->paramNames(false);
    int numPlanesUsed = 0; // number of planes used

    // Iterate over childless properties with names starting with maskPlanePrefix
    for (NameList::const_iterator i = paramNames.begin(); i != paramNames.end(); ++i) {
        if (i->compare(0, maskPlanePrefix.size(), maskPlanePrefix) == 0) {
            // split off maskPlanePrefix to obtain plane name
            std::string planeName = i->substr(maskPlanePrefix.size());
            int const planeId = metadata->getAsInt(*i);

            MaskPlaneDict::const_iterator plane = newDict.find(planeName);
            if (plane != newDict.end() && planeId != plane->second) {
               throw LSST_EXCEPT(pexExcept::RuntimeErrorException,
                                 "File specifies plane " + planeName + " twice"); 
            }
            for (MaskPlaneDict::const_iterator j = newDict.begin(); j != newDict.end(); ++j) {
                if (planeId == j->second) {
                    throw LSST_EXCEPT(pexExcept::RuntimeErrorException,
                                      str(boost::format("File specifies plane %s has same value (%d) as %s") %
                                          planeName % planeId % j->first));
                }
            }
            // build new entry
            if (numPlanesUsed >= getNumPlanesMax()) {
                // Max number of planes already allocated
                throw LSST_EXCEPT(pexExcept::RuntimeErrorException,
                                  str(boost::format("Max number of planes (%1%) already used") %
                                      getNumPlanesMax()));
            }
            newDict[planeName] = planeId; 
        }
    }
    return newDict;
}

/**
 * \brief print the mask plane dictionary to std::cout
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::printMaskPlanes() const
{
    _maskDict->print();
}

/*
 * Static members of Mask
 */
template<typename MaskPixelT>
std::string const Mask<MaskPixelT>::maskPlanePrefix("MP_");

template<typename MaskPixelT>
PTR(detail::MaskDict) Mask<MaskPixelT>::_maskPlaneDict()
{
    return detail::MaskDict::makeMaskDict();
}

//
// Explicit instantiations
//
template class Mask<MaskPixel>;
}}}
