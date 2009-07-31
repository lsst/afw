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
#include "lsst/afw/image/Wcs.h"
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
namespace ex = lsst::pex::exceptions;
namespace logging = lsst::pex::logging;

using lsst::daf::base::PropertySet;

///
/// initialise mask planes; called by ctors
template<typename MaskPixelT>
void image::Mask<MaskPixelT>::_initializePlanes(MaskPlaneDict const& planeDefs) {
    logging::Trace("afw.Mask", 5,
                   boost::format("Number of mask planes: %d") % getNumPlanesMax());

    if (planeDefs.size() > 0 && planeDefs != _maskPlaneDict) {
        _maskPlaneDict = planeDefs;
        _myMaskDictVersion = ++_maskDictVersion;
    }
}

/// Constructor of uninitialised mask
template<typename MaskPixelT>
image::Mask<MaskPixelT>::Mask(int width, ///< Number of columns
                              int height, ///< Number of rows
                              MaskPlaneDict const& planeDefs ///< desired mask planes
                             ) :
    image::ImageBase<MaskPixelT>(width, height),
    _myMaskDictVersion(_maskDictVersion) {
    _initializePlanes(planeDefs);
}

/// Constructor of initialised mask
template<typename MaskPixelT>
image::Mask<MaskPixelT>::Mask(int width, ///< Number of columns
                              int height, ///< Number of rows
                              MaskPixelT initialValue, ///< Initial value
                              MaskPlaneDict const& planeDefs ///< desired mask planes
                             ) :
    image::ImageBase<MaskPixelT>(width, height),
    _myMaskDictVersion(_maskDictVersion) {
    _initializePlanes(planeDefs);
    *this = initialValue;
}

/// Constructor of uninitialised mask
template<typename MaskPixelT>
image::Mask<MaskPixelT>::Mask(const std::pair<int, int> dimensions, ///< Desired number of columns/rows
                              MaskPlaneDict const& planeDefs ///< desired mask planes
                             ) :
    image::ImageBase<MaskPixelT>(dimensions),
    _myMaskDictVersion(_maskDictVersion) {
    _initializePlanes(planeDefs);
}

/// Constructor of uninitialised mask
template<typename MaskPixelT>
image::Mask<MaskPixelT>::Mask(const std::pair<int, int> dimensions, ///< Desired number of columns/rows
                              MaskPixelT initialValue, ///< Initial value
                              MaskPlaneDict const& planeDefs ///< desired mask planes
                             ) :
    image::ImageBase<MaskPixelT>(dimensions),
    _myMaskDictVersion(_maskDictVersion) {
    _initializePlanes(planeDefs);
    *this = initialValue;
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
        lsst::daf::base::PropertySet::Ptr metadata,        //!< file metadata (may point to NULL)
        BBox const& bbox,                                  //!< Only read these pixels
        bool const conformMasks                            //!< Make Mask conform to mask layout in file?
                             ) :
    image::ImageBase<MaskPixelT>(),
    _myMaskDictVersion(_maskDictVersion) {
    
    if (!metadata) {
        metadata = PropertySet::Ptr(new PropertySet()); //TODOsmm createPropertyNode("FitsMetadata");
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
        throw LSST_EXCEPT(ex::NotFoundException,
                          (boost::format("File %s doesn't exist") % fileName).str());
    }

    if (!metadata) {
        metadata = lsst::daf::base::PropertySet::Ptr(new lsst::daf::base::PropertySet);
    }

    if (!image::fits_read_image<fits_mask_types>(fileName, *_getRawImagePtr(), metadata, hdu, bbox)) {
        throw LSST_EXCEPT(image::FitsException, (boost::format("Failed to read %s HDU %d") % fileName % hdu).str());
    }
    _setRawView();

    if (bbox) {
        this->setXY0(bbox.getLLC());
    }
    /*
     * We will interpret one of the header WCSs as providing the (X0, Y0) values
     */
    this->setXY0(this->getXY0() + image::detail::getImageXY0FromMetadata(image::detail::wcsNameForXY0, metadata.get()));
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

/**
 * Write a Mask to the specified file
 */
template<typename MaskPixelT>
void image::Mask<MaskPixelT>::writeFits(
        std::string const& fileName, ///< File to write
        boost::shared_ptr<const lsst::daf::base::PropertySet> metadata_i, //!< metadata to write to header; or NULL
        std::string const& mode    ///< "w" to write a new file; "a" to append
                                       ) const {

    lsst::daf::base::PropertySet::Ptr metadata;
    if (metadata_i) {
        metadata = metadata_i->deepCopy();
    } else {
        metadata = lsst::daf::base::PropertySet::Ptr(new lsst::daf::base::PropertySet());
    }
    addMaskPlanesToMetadata(metadata);
    //
    // Add WCS with (X0, Y0) information
    //
    PropertySet::Ptr wcsAMetadata = image::detail::createTrivialWcsAsPropertySet(image::detail::wcsNameForXY0,
                                                                                 this->getX0(), this->getY0());
    metadata->combine(wcsAMetadata);

    image::fits_write_view(fileName, _getRawView(), metadata, mode);
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
        throw LSST_EXCEPT(ex::RuntimeErrorException,
                          (boost::format("Max number of planes (%1%) already used") % getNumPlanesMax()).str());
    }
}

// This is a private function.  It sets the plane of the given planeId to be name
// with minimal checking.   Mainly used by setMaskPlaneMetadata

template<typename MaskPixelT>
int image::Mask<MaskPixelT>::addMaskPlane(std::string name, int planeId)
{
    if (planeId < 0 || planeId >= getNumPlanesMax()) {
        throw LSST_EXCEPT(ex::RangeErrorException,
                          (boost::format("mask plane id must be between 0 and %1%") % (getNumPlanesMax() - 1)).str());
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
        logging::Trace("afw.Mask", 0,
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
// @throw lsst::pex::exceptions::InvalidParameterException if plane is invalid
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
    throw LSST_EXCEPT(ex::InvalidParameterException, (boost::format("Invalid mask plane: %d") % plane).str());
}

template<typename MaskPixelT>
int image::Mask<MaskPixelT>::getMaskPlane(const std::string& name) {
    const int plane = getMaskPlaneNoThrow(name);
    
    if (plane < 0) {
        throw LSST_EXCEPT(ex::InvalidParameterException, (boost::format("Invalid mask plane: %s") % name).str());
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
namespace bl = boost::lambda;

/**
 * @brief OR a bitmask into a Mask
 */
template<typename MaskPixelT>
void image::Mask<MaskPixelT>::operator|=(const MaskPixelT val) {
    transform_pixels(_getRawView(), _getRawView(), bl::ret<MaskPixelT>(bl::_1 | val));
}

/**
 * @brief OR a Mask into a Mask
 */
template<typename MaskPixelT>
void image::Mask<MaskPixelT>::operator|=(const Mask& rhs) {
    checkMaskDictionaries(rhs);

    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("Images are of different size, %dx%d v %dx%d") %
                           this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()).str());
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), bl::ret<MaskPixelT>(bl::_1 | bl::_2));
}

/**
 * @brief AND a bitmask into a Mask
 */
template<typename MaskPixelT>
void image::Mask<MaskPixelT>::operator&=(const MaskPixelT val) {
    transform_pixels(_getRawView(), _getRawView(), bl::ret<MaskPixelT>(bl::_1 & val));
}

/**
 * @brief AND a Mask into a Mask
 */
template<typename MaskPixelT>
void image::Mask<MaskPixelT>::operator&=(const Mask& rhs) {
    checkMaskDictionaries(rhs);

    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("Images are of different size, %dx%d v %dx%d") %
                           this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()).str());
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), bl::ret<MaskPixelT>(bl::_1 & bl::_2));
}

/**
 * @brief XOR a bitmask into a Mask
 */
template<typename MaskPixelT>
void image::Mask<MaskPixelT>::operator^=(const MaskPixelT val) {
    transform_pixels(_getRawView(), _getRawView(), bl::ret<MaskPixelT>(bl::_1 ^ val));
}

/**
 * @brief XOR a Mask into a Mask
 */
template<typename MaskPixelT>
void image::Mask<MaskPixelT>::operator^=(const Mask& rhs) {
    checkMaskDictionaries(rhs);

    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("Images are of different size, %dx%d v %dx%d") %
                           this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()).str());
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), bl::ret<MaskPixelT>(bl::_1 ^ bl::_2));
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
 * @brief Given a PropertySet, replace any existing MaskPlane assignments with the current ones.
 */
template<typename MaskPixelT>
void image::Mask<MaskPixelT>::addMaskPlanesToMetadata(lsst::daf::base::PropertySet::Ptr metadata) {
    if (!metadata) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Null PropertySet::Ptr");
    }

    // First, clear existing MaskPlane metadata
    typedef std::vector<std::string> NameList;
    NameList paramNames = metadata->paramNames(false);
    for (NameList::const_iterator i = paramNames.begin(); i != paramNames.end(); ++i) {
        if (i->compare(0, maskPlanePrefix.size(), maskPlanePrefix) == 0) {
            metadata->remove(*i);
        }
    }

    // Add new MaskPlane metadata
    for (MaskPlaneDict::iterator i = _maskPlaneDict.begin(); i != _maskPlaneDict.end() ; ++i) {
        std::string const planeName = i->first;
        int const planeNumber = i->second;

        if (planeName != "") {
            metadata->add(maskPlanePrefix + planeName, planeNumber);
        }
    }
}


/**
 * @brief Given a PropertySet that contains the MaskPlane assignments, setup the MaskPlanes.
 *
 * @returns a dictionary of mask names/plane assignments
 */
template<typename MaskPixelT>
typename image::Mask<MaskPixelT>::MaskPlaneDict image::Mask<MaskPixelT>::parseMaskPlaneMetadata(
	lsst::daf::base::PropertySet::Ptr const metadata //!< metadata from a Mask
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
               throw LSST_EXCEPT(ex::RuntimeErrorException, "File specifies plane " + planeName + " twice"); 
            }
            for (MaskPlaneDict::const_iterator i = newDict.begin(); i != newDict.end(); ++i) {
                if (planeId == i->second) {
                    throw LSST_EXCEPT(ex::RuntimeErrorException,
                                      (boost::format("File specifies plane %s has same value (%d) as %s") %
                                          planeName % planeId % i->first).str());
                }
            }
            // build new entry
            if (numPlanesUsed >= getNumPlanesMax()) {
                // Max number of planes already allocated
                throw LSST_EXCEPT(ex::RuntimeErrorException,
                                  (boost::format("Max number of planes (%1%) already used") %
                                      getNumPlanesMax()).str());
            }
            newDict[planeName] = planeId; 
        }
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
    planeDict["SAT"] = ++i;             // should be SATURATED
    planeDict["INTRP"] = ++i;           // should be INTERPOLATED
    planeDict["CR"] = ++i;
    planeDict["EDGE"] = ++i;
    planeDict["DETECTED"] = ++i;
    planeDict["DETECTED_NEGATIVE"] = ++i;

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
