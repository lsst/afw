// -*- lsst-c++ -*-
/// \file
/// \brief Implementations of Mask class methods

#include <list>
#include <string>

#include "boost/lambda/lambda.hpp"
#include "boost/format.hpp"
#include "boost/filesystem/path.hpp"

#include "lsst/daf/base.h"
#include "lsst/daf/data/LsstBase.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image/Mask.h"

#include "lsst/afw/image/LsstImageTypes.h"

/************************************************************************************************************/
//
// for FITS code
//
#include <boost/mpl/vector.hpp>
#include "boost/gil/gil_all.hpp"
#include "lsst/afw/image/fits/fits_io.h"
#include "lsst/afw/image/fits/fits_io_mpl.h"

namespace image = lsst::afw::image;

template<typename MaskPixelT>
image::Mask<MaskPixelT>::Mask(int width, int height, MaskPlaneDict const& planeDefs) :
    image::ImageBase<MaskPixelT>(width, height),
    _myMaskDictVersion(_maskDictVersion) {

    lsst::pex::logging::Trace("afw.Mask", 5,
              boost::format("Number of mask planes: %d") % getNumPlanesMax());

    if (planeDefs.size() > 0 && planeDefs != _maskPlaneDict) {
        _maskPlaneDict = planeDefs;
        _myMaskDictVersion = ++_maskDictVersion;
    }
}

template<typename MaskPixelT>
image::Mask<MaskPixelT>::Mask(const std::pair<int, int> dimensions, MaskPlaneDict const& planeDefs) :
    image::ImageBase<MaskPixelT>(dimensions),
    _myMaskDictVersion(_maskDictVersion) {

    lsst::pex::logging::Trace("afw.Mask", 5,
              boost::format("Number of mask planes: %d") % getNumPlanesMax());

    if (planeDefs.size() > 0 && planeDefs != _maskPlaneDict) {
        _maskPlaneDict = planeDefs;
        _myMaskDictVersion = ++_maskDictVersion;
    }
}

template<typename MaskPixelT>
image::Mask<MaskPixelT>::Mask(Mask const& rhs, const BBox& bbox, const bool deep) :
    image::ImageBase<MaskPixelT>(rhs, bbox, deep),
    _myMaskDictVersion(rhs._myMaskDictVersion) {
}

template<typename MaskPixelT>
image::Mask<MaskPixelT>::Mask(image::Mask<MaskPixelT> const& rhs, bool deep) :
    image::ImageBase<MaskPixelT>(rhs, deep),
    _myMaskDictVersion(rhs._myMaskDictVersion) {
}

/************************************************************************************************************/

template<typename PixelT>
void image::Mask<PixelT>::swap(Mask &rhs) {
    using std::swap;                    // See Meyers, Effective C++, Item 25

    ImageBase<PixelT>::swap(rhs);
    swap(_myMaskDictVersion, rhs._myMaskDictVersion);    
}

template<typename PixelT>
void image::swap(Mask<PixelT>& a, Mask<PixelT>& b) {
    a.swap(b);
}

/*
 *
 */
template<typename MaskPixelT>
image::Mask<MaskPixelT>& image::Mask<MaskPixelT>::operator=(const image::Mask<MaskPixelT>& rhs) {
    Mask tmp(rhs);
    swap(tmp);                        // See Meyers, Effective C++, Item 11
    
    return *this;
}

template<typename MaskPixelT>
image::Mask<MaskPixelT>& image::Mask<MaskPixelT>::operator=(const MaskPixelT rhs) {
    fill_pixels(_getRawView(), rhs);

    return *this;
}

/**
 * @brief Create a Mask from a FITS file on disk
 *
 * The meaning of the bitplanes is given in the header.  If conformMasks is false (default),
 * the bitvalues will be changed to match those in Mask's plane dictionary.  If it's true, the
 * bitvalues will be left alone, but Mask's dictionary will be modified to match the
 * on-disk version
 */
template<typename MaskPixelT>
image::Mask<MaskPixelT>::Mask(std::string const& fileName, //!< Name of file to read
        int const hdu,                                     //!< HDU to read 
        lsst::daf::base::DataProperty::PtrType metadata,   //!< file metadata (may point to NULL)
        bool const conformMasks                            //!< Make Mask conform to mask layout in file?
                             ) :
    image::ImageBase<MaskPixelT>() {

    if (metadata.get() == NULL) {
        metadata = lsst::daf::base::DataProperty::createPropertyNode("FitsMetadata");
    }
    //
    // These are the permitted input file types
    //
    typedef boost::mpl::vector<
        lsst::afw::image::detail::types_traits<unsigned char>::image_t,
        lsst::afw::image::detail::types_traits<unsigned short>::image_t,
        lsst::afw::image::detail::types_traits<short>::image_t
    > fits_mask_types;

    if (!boost::filesystem::exists(fileName)) {
        throw lsst::pex::exceptions::NotFound(boost::format("File %s doesn't exist") % fileName);
    }

    if (!image::fits_read_image<fits_mask_types>(fileName, *_getRawImagePtr(), metadata)) {
        throw lsst::pex::exceptions::FitsError(boost::format("Failed to read %s HDU %d") % fileName % hdu);
    }
    _setRawView();
    //
    // OK, we've read it.  Now make sense of its mask planes
    //
    MaskPlaneDict fileMaskDict = parseMaskPlaneMetadata(metadata); // look for mask planes in the file

    if (fileMaskDict == _maskPlaneDict) { // file is consistent with Mask
        return;
    }
    
    if (conformMasks) {                 // adopt the definitions in the file
        if (_maskPlaneDict != fileMaskDict) {
            _maskPlaneDict = fileMaskDict;
            _maskDictVersion++;
        }
    }

    conformMaskPlanes(fileMaskDict);    // convert planes defined by fileMaskDict to the order
    ;                                   // defined by Mask::_maskPlaneDict
}

template<typename MaskPixelT>
void image::Mask<MaskPixelT>::writeFits(std::string const& fileName) const {
    lsst::daf::base::DataProperty::PtrType metadata = lsst::daf::base::DataProperty::createPropertyNode("FitsMetadata");
    addMaskPlanesToMetadata(metadata);
    
    image::fits_write_view(fileName, _getRawView(), metadata);
}

template<typename MaskPixelT>
int image::Mask<MaskPixelT>::addMaskPlane(const std::string& name)
{
    int const id = getMaskPlaneNoThrow(name);

    if (id >= 0) {
        return id;
    }
    // build new entry
    int const numPlanesUsed = _maskPlaneDict.size();
    if (numPlanesUsed < getNumPlanesMax()) {
        _maskPlaneDict[name] = numPlanesUsed;
        
        return _maskPlaneDict[name];
    } else {
        // Max number of planes already allocated
        throw lsst::pex::exceptions::Runtime("Max number of planes already used")
            << lsst::daf::base::DataProperty("numPlanesUsed", _maskPlaneDict.size())
            << lsst::daf::base::DataProperty("numPlanesMax", getNumPlanesMax());
    }
}

// This is a private function.  It sets the plane of the given planeId to be name
// with minimal checking.   Mainly used by setMaskPlaneMetadata

template<typename MaskPixelT>
int image::Mask<MaskPixelT>::addMaskPlane(std::string name, int planeId)
{
    if (planeId < 0 || planeId >= getNumPlanesMax()) {
        throw;
    }

    _maskPlaneDict[name] = planeId;

    return planeId;
}

template<typename MaskPixelT>
void image::Mask<MaskPixelT>::removeMaskPlane(const std::string& name)
{
     int id;
     try {
        id = getMaskPlane(name);
        clearMaskPlane(id);
        _maskPlaneDict.erase(name);
        _myMaskDictVersion = ++_maskDictVersion;
        return;
     } catch (std::exception &e) {
        lsst::pex::logging::Trace("afw.Mask", 0,
                                  boost::format("%s Plane %s not present in this Mask") % e.what() % name);
        return;
     }
     
}

// \brief Return the bitmask corresponding to plane, or 0 if invalid
template<typename MaskPixelT>
MaskPixelT image::Mask<MaskPixelT>::getBitMaskNoThrow(int plane) {
    return (plane >= 0 && plane < getNumPlanesMax()) ? (1 << plane) : 0;
}

// \brief Return the bitmask corresponding to plane
//
// @throw lsst::pex::exceptions::InvalidParameter if plane is invalid
template<typename MaskPixelT>
MaskPixelT image::Mask<MaskPixelT>::getBitMask(int plane) {
    for (MaskPlaneDict::const_iterator i = _maskPlaneDict.begin(); i != _maskPlaneDict.end(); ++i) {
        if (plane == i->second) {
            MaskPixelT const bitmask = getBitMaskNoThrow(plane);
            if (bitmask == 0) {         // failed
                break;
            }
            return bitmask;
        }
    }
    throw lsst::pex::exceptions::InvalidParameter(boost::format("Invalid mask plane: %d") % plane);
}

template<typename MaskPixelT>
int image::Mask<MaskPixelT>::getMaskPlane(const std::string& name) {
    const int plane = getMaskPlaneNoThrow(name);
    
    if (plane < 0) {
        throw lsst::pex::exceptions::InvalidParameter(boost::format("Invalid mask plane: %s") % name);
    } else {
        return plane;
    }
}

template<typename MaskPixelT>
int image::Mask<MaskPixelT>::getMaskPlaneNoThrow(const std::string& name) {
    MaskPlaneDict::const_iterator plane = _maskPlaneDict.find(name);
    
    if (plane == _maskPlaneDict.end()) {
        return -1;
    } else {
        return plane->second;
    }
}

template<typename MaskPixelT>
MaskPixelT image::Mask<MaskPixelT>::getPlaneBitMask(const std::string& name) {
    return getBitMask(getMaskPlane(name));
}

// \brief Reset the maskPlane dictionary
template<typename MaskPixelT>
void image::Mask<MaskPixelT>::clearMaskPlaneDict() {
    _maskPlaneDict.clear();
    _myMaskDictVersion = ++_maskDictVersion;
}

// \brief Clear all the pixels in a Mask
template<typename MaskPixelT>
void image::Mask<MaskPixelT>::clearAllMaskPlanes() {
    *this = 0;
}

// clearMaskPlane(int plane) clears the bit specified by "plane" in all pixels in the mask
//
template<typename MaskPixelT>
void image::Mask<MaskPixelT>::clearMaskPlane(int plane) {
    *this &= ~getBitMask(plane);
}

// \brief Adjust this mask to conform to the standard Mask class's mask plane dictionary,
// adding any new mask planes to the standard.
//
// Ensures that this mask (presumably from some external source) has the same plane assignments
// as the Mask class. If a change in plane assignments is needed, the bits within each pixel
// are permuted as required.
//
// Any new mask planes found in this mask are added to unused slots in the Mask class's mask plane dictionary.
//
template<typename MaskPixelT>
void image::Mask<MaskPixelT>::conformMaskPlanes(
    const MaskPlaneDict& currentPlaneDict   ///< mask plane dictionary for this mask
) {

    if (_maskPlaneDict == currentPlaneDict) {
        _myMaskDictVersion = _maskDictVersion;
        return;   // nothing to do
    }
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
            for (typename Mask::x_iterator ptr = this->row_begin(r), end = this->row_end(r); ptr != end; ++ptr) {
                MaskPixelT const pixel = *ptr;

                MaskPixelT newPixel = pixel & keepBitmask; // value of invariant mask bits
                for (int i = 0; i < numReMap; i++) {
                    if (pixel & currentMask[i]) newPixel |= canonicalMask[i];
                }

                *ptr = newPixel;
            }
        }
    }
    // We've made the planes match the current mask dictionary
    _myMaskDictVersion = _maskDictVersion;
}

/************************************************************************************************************/

template<typename MaskPixelT>
typename image::ImageBase<MaskPixelT>::PixelReference image::Mask<MaskPixelT>::operator()(int x, int y) {
    return this->ImageBase<MaskPixelT>::operator()(x, y);
}

template<typename MaskPixelT>
typename image::ImageBase<MaskPixelT>::PixelConstReference image::Mask<MaskPixelT>::operator()(int x, int y) const {
    return this->ImageBase<MaskPixelT>::operator()(x, y);
}

template<typename MaskPixelT>
bool image::Mask<MaskPixelT>::operator()(int x, int y, int plane) const {
    return !!(this->ImageBase<MaskPixelT>::operator()(x, y) & getBitMask(plane)); // ! ! converts an int to a bool
}

/************************************************************************************************************/
//
// N.b. We could use the STL, but I find boost::lambda clearer, and more easily extended
// to e.g. setting random numbers
//    transform_pixels(_getRawView(), _getRawView(), lambda::ret<PixelT>(lambda::_1 + val));
// is equivalent to
//    transform_pixels(_getRawView(), _getRawView(), std::bind2nd(std::plus<PixelT>(), val));
//
using boost::lambda::ret;
using boost::lambda::_1;
using boost::lambda::_2;

/**
 * @brief OR a bitmask into a Mask
 */
template<typename MaskPixelT>
void image::Mask<MaskPixelT>::operator|=(const MaskPixelT val) {
    transform_pixels(_getRawView(), _getRawView(), ret<MaskPixelT>(_1 | val));
}

/**
 * @brief OR a Mask into a Mask
 */
template<typename MaskPixelT>
void image::Mask<MaskPixelT>::operator|=(const Mask& rhs) {
    checkMaskDictionaries(rhs);

    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), ret<MaskPixelT>(_1 | _2));
}

/**
 * @brief AND a bitmask into a Mask
 */
template<typename MaskPixelT>
void image::Mask<MaskPixelT>::operator&=(const MaskPixelT val) {
    transform_pixels(_getRawView(), _getRawView(), ret<MaskPixelT>(_1 & val));
}

/**
 * @brief AND a Mask into a Mask
 */
template<typename MaskPixelT>
void image::Mask<MaskPixelT>::operator&=(const Mask& rhs) {
    checkMaskDictionaries(rhs);

    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), ret<MaskPixelT>(_1 & _2));
}

/**
 * @brief Set the bit specified by "plane" for pixels (x0, y) ... (x1, y)
 */
template<typename MaskPixelT>
void image::Mask<MaskPixelT>::setMaskPlaneValues(const int plane, const int x0, const int x1, const int y) {
    MaskPixelT const bitMask = getBitMask(plane);
    
    for (int x = x0; x <= x1; x++) {
        operator()(x, y) = operator()(x, y) | bitMask;
    }
}

/**
 * @brief Given a DataProperty, replace any existing MaskPlane assignments with the current ones.
 *
 * @throw Throws lsst::pex::exceptions::InvalidParameter if given DataProperty is not a node
 */
template<typename MaskPixelT>
void image::Mask<MaskPixelT>::addMaskPlanesToMetadata(lsst::daf::base::DataProperty::PtrType rootPtr) {
     if( rootPtr->isNode() != true ) {
        throw lsst::pex::exceptions::InvalidParameter( "Given DataProperty object is not a node" );
        
     }

    // First, clear existing MaskPlane metadata
    rootPtr->deleteAll( maskPlanePrefix +".*", false );

    // Add new MaskPlane metadata
    for (MaskPlaneDict::iterator i = _maskPlaneDict.begin(); i != _maskPlaneDict.end() ; ++i) {
        std::string const planeName = i->first;
        int const planeNumber = i->second;
        
        if (planeName != "") {
            rootPtr->addProperty(
                lsst::daf::base::DataProperty::PtrType(new lsst::daf::base::DataProperty(Mask::maskPlanePrefix + planeName, planeNumber)));
        }
    }
}


/**
 * @brief Given a DataProperty that contains the MaskPlane assignments setup the MaskPlanes.
 *
 * @returns a dictionary of mask names/plane assignments
 */
template<typename MaskPixelT>
typename image::Mask<MaskPixelT>::MaskPlaneDict image::Mask<MaskPixelT>::parseMaskPlaneMetadata(
	lsst::daf::base::DataProperty::PtrType const rootPtr //!< metadata from a Mask
                                                                                               ) {
    MaskPlaneDict newDict;

    lsst::daf::base::DataProperty::iteratorRangeType range = rootPtr->searchAll( maskPlanePrefix +".*" );
    if (std::distance(range.first, range.second) == 0) {
        return newDict;
    }

    int numPlanesUsed = 0;              // number of planes used
    // Iterate through matching keyWords setting the dictionary
    lsst::daf::base::DataProperty::ContainerIteratorType iter;
    for( iter = range.first; iter != range.second; ++iter, ++numPlanesUsed ) {
        lsst::daf::base::DataProperty::PtrType dpPtr = *iter;
        // split off the "MP_" to get the planeName
        std::string const keyWord = dpPtr->getName();
        std::string const planeName = keyWord.substr(maskPlanePrefix.size());
        // will throw an exception if the found item does not contain const int
        int const planeId = boost::any_cast<const int>(dpPtr->getValue());

        MaskPlaneDict::const_iterator plane = newDict.find(planeName);
        if (plane != newDict.end() && planeId != plane->second) {
            throw lsst::pex::exceptions::Runtime("File specifies plane " + planeName + " twice");
        }

        for (MaskPlaneDict::const_iterator i = newDict.begin(); i != newDict.end(); ++i) {
            if (planeId == i->second) {
                throw lsst::pex::exceptions::Runtime(boost::format("File specifies plane %s has same value (%d) as %s") %
                                                     planeName % planeId % i->first);
            }
        }

        // build new entry
        if (numPlanesUsed >= getNumPlanesMax()) {
            // Max number of planes already allocated
            throw lsst::pex::exceptions::Runtime("Max number of planes already used")
                << lsst::daf::base::DataProperty("numPlanesUsed", numPlanesUsed)
                << lsst::daf::base::DataProperty("numPlanesMax", getNumPlanesMax());
        }

        newDict[planeName] = planeId;
        
    }

    return newDict;
}

template<typename MaskPixelT>
void image::Mask<MaskPixelT>::printMaskPlanes() {
    for (MaskPlaneDict::const_iterator i = _maskPlaneDict.begin(); i != _maskPlaneDict.end() ; i++) {
        std::string const planeName = i->first;
        int const planeNumber = i->second;

        std::cout << "Plane " << planeNumber << " -> " << planeName << std::endl;
    }
}

/*
 * Default Mask planes
 */
template<typename MaskPixelT>
static typename image::Mask<MaskPixelT>::MaskPlaneDict initMaskPlanes() {
    typename image::Mask<MaskPixelT>::MaskPlaneDict planeDict = typename image::Mask<MaskPixelT>::MaskPlaneDict();

    int i = -1;
    planeDict["BAD"] = ++i;
    planeDict["SAT"] = ++i;
    planeDict["INTRP"] = ++i;
    planeDict["CR"] = ++i;
    planeDict["EDGE"] = ++i;

    return planeDict;
}

/*
 * Static members of Mask
 */
template<typename MaskPixelT>
std::string const image::Mask<MaskPixelT>::maskPlanePrefix("MP_");

template<typename MaskPixelT>
typename image::Mask<MaskPixelT>::MaskPlaneDict image::Mask<MaskPixelT>::_maskPlaneDict = initMaskPlanes<MaskPixelT>();

template<typename MaskPixelT>
int image::Mask<MaskPixelT>::_maskDictVersion = 0;    // version number for bitplane dictionary

//
// Explicit instantiations
//
template class image::Mask<image::MaskPixel>;
