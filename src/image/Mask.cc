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
#include "boost/mpl/vector.hpp"
#include "boost/gil/gil_all.hpp"
#include "lsst/afw/image/fits/fits_io.h"
#include "lsst/afw/image/fits/fits_io_mpl.h"

namespace afwImage = lsst::afw::image;
namespace dafBase = lsst::daf::base;
namespace pexExcept = lsst::pex::exceptions;
namespace pexLog = lsst::pex::logging;

/**
 * \brief Initialise mask planes; called by constructors
 */
template<typename MaskPixelT>
void afwImage::Mask<MaskPixelT>::_initializePlanes(MaskPlaneDict const& planeDefs) {
    pexLog::Trace("afw.Mask", 5,
                   boost::format("Number of mask planes: %d") % getNumPlanesMax());

    if (planeDefs.size() > 0 && planeDefs != _maskPlaneDict) {
        _maskPlaneDict = planeDefs;
        _myMaskDictVersion = ++_maskDictVersion;
    }
}

/**
 * \brief Construct a Mask initialized to 0x0
 */
template<typename MaskPixelT>
afwImage::Mask<MaskPixelT>::Mask(
    int width, ///< Number of columns
    int height, ///< Number of rows
    MaskPlaneDict const& planeDefs ///< desired mask planes
) :
    afwImage::ImageBase<MaskPixelT>(width, height),
    _myMaskDictVersion(_maskDictVersion) {
    _initializePlanes(planeDefs);
    *this = 0x0;
}

/**
 * \brief Construct a Mask initialized to a specified value
 */
template<typename MaskPixelT>
afwImage::Mask<MaskPixelT>::Mask(
    int width, ///< Number of columns
    int height, ///< Number of rows
    MaskPixelT initialValue, ///< Initial value
    MaskPlaneDict const& planeDefs ///< desired mask planes
) :
    afwImage::ImageBase<MaskPixelT>(width, height),
    _myMaskDictVersion(_maskDictVersion) {
    _initializePlanes(planeDefs);
    *this = initialValue;
}

/**
 * \brief Construct a Mask initialized to 0x0
 */
template<typename MaskPixelT>
afwImage::Mask<MaskPixelT>::Mask(
    const std::pair<int, int> dimensions, ///< Desired number of columns/rows
    MaskPlaneDict const& planeDefs ///< desired mask planes
) :
    afwImage::ImageBase<MaskPixelT>(dimensions),
    _myMaskDictVersion(_maskDictVersion) {
    _initializePlanes(planeDefs);
    *this = 0x0;
}

/**
 * \brief Construct a Mask initialized to a specified value
 */
template<typename MaskPixelT>
afwImage::Mask<MaskPixelT>::Mask(
    const std::pair<int, int> dimensions, ///< Desired number of columns/rows
    MaskPixelT initialValue, ///< Initial value
    MaskPlaneDict const& planeDefs ///< desired mask planes
) :
    afwImage::ImageBase<MaskPixelT>(dimensions),
    _myMaskDictVersion(_maskDictVersion) {
    _initializePlanes(planeDefs);
    *this = initialValue;
}

/**
 * \brief Construct a Mask from a subregion of another Mask
 */
template<typename MaskPixelT>
afwImage::Mask<MaskPixelT>::Mask(
    Mask const &rhs,    ///< mask to copy
    BBox const &bbox,   ///< subregion to copy
    bool const deep     ///< deep copy? (construct a view with shared pixels if false)
) :
    afwImage::ImageBase<MaskPixelT>(rhs, bbox, deep),
    _myMaskDictVersion(rhs._myMaskDictVersion) {
}

/**
 * \brief Construct a Mask from another Mask
 */
template<typename MaskPixelT>
afwImage::Mask<MaskPixelT>::Mask(
    Mask const& rhs,    ///< mask to copy
    bool deep           ///< deep copy? (construct a view with shared pixels if false)
) :
    afwImage::ImageBase<MaskPixelT>(rhs, deep),
    _myMaskDictVersion(rhs._myMaskDictVersion) {
}

/************************************************************************************************************/

template<typename PixelT>
void afwImage::Mask<PixelT>::swap(Mask &rhs) {
    using std::swap;                    // See Meyers, Effective C++, Item 25

    ImageBase<PixelT>::swap(rhs);
    swap(_myMaskDictVersion, rhs._myMaskDictVersion);    
}

template<typename PixelT>
void afwImage::swap(Mask<PixelT>& a, Mask<PixelT>& b) {
    a.swap(b);
}

template<typename MaskPixelT>
afwImage::Mask<MaskPixelT>& afwImage::Mask<MaskPixelT>::operator=(const afwImage::Mask<MaskPixelT>& rhs) {
    Mask tmp(rhs);
    swap(tmp);                        // See Meyers, Effective C++, Item 11
    
    return *this;
}

template<typename MaskPixelT>
afwImage::Mask<MaskPixelT>& afwImage::Mask<MaskPixelT>::operator=(MaskPixelT const rhs) {
    fill_pixels(_getRawView(), rhs);

    return *this;
}

/**
 * \brief Create a Mask from a FITS file on disk
 *
 * The meaning of the bitplanes is given in the header.  If conformMasks is false (default),
 * the bitvalues will be changed to match those in Mask's plane dictionary.  If it's true, the
 * bitvalues will be left alone, but Mask's dictionary will be modified to match the
 * on-disk version
 */
template<typename MaskPixelT>
afwImage::Mask<MaskPixelT>::Mask(std::string const& fileName, ///< Name of file to read
        int const hdu,                                     ///< HDU to read 
        lsst::daf::base::PropertySet::Ptr metadata,        ///< file metadata (may point to NULL)
        BBox const& bbox,                                  ///< Only read these pixels
        bool const conformMasks                            ///< Make Mask conform to mask layout in file?
) :
    afwImage::ImageBase<MaskPixelT>(),
    _myMaskDictVersion(_maskDictVersion) {
    
    if (!metadata) {
        //TODOsmm createPropertyNode("FitsMetadata");
        metadata = dafBase::PropertySet::Ptr(new dafBase::PropertySet()); 
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
        throw LSST_EXCEPT(pexExcept::NotFoundException,
                          (boost::format("File %s doesn't exist") % fileName).str());
    }

    if (!metadata) {
        metadata = dafBase::PropertySet::Ptr(new dafBase::PropertySet);
    }

    if (!afwImage::fits_read_image<fits_mask_types>(fileName, *_getRawImagePtr(), metadata, hdu, bbox)) {
        throw LSST_EXCEPT(afwImage::FitsException,
            (boost::format("Failed to read %s HDU %d") % fileName % hdu).str());
    }
    _setRawView();

    if (bbox) {
        this->setXY0(bbox.getLLC());
    }
    /*
     * We will interpret one of the header WCSs as providing the (X0, Y0) values
     */
    this->setXY0(this->getXY0() + afwImage::detail::getImageXY0FromMetadata(afwImage::detail::wcsNameForXY0,
                                                                            metadata.get()));
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
 * \brief Write a Mask to the specified file
 */
template<typename MaskPixelT>
void afwImage::Mask<MaskPixelT>::writeFits(
    std::string const& fileName, ///< File to write
    boost::shared_ptr<const lsst::daf::base::PropertySet> metadata_i, ///< metadata to write to header,
        ///< or a null pointer if none
    std::string const& mode    ///< "w" to write a new file; "a" to append
) const {

    dafBase::PropertySet::Ptr metadata;
    if (metadata_i) {
        metadata = metadata_i->deepCopy();
    } else {
        metadata = dafBase::PropertySet::Ptr(new dafBase::PropertySet());
    }
    addMaskPlanesToMetadata(metadata);
    //
    // Add WCS with (X0, Y0) information
    //
    dafBase::PropertySet::Ptr wcsAMetadata = afwImage::detail::createTrivialWcsAsPropertySet(
        afwImage::detail::wcsNameForXY0, this->getX0(), this->getY0());
    metadata->combine(wcsAMetadata);

    afwImage::fits_write_view(fileName, _getRawView(), metadata, mode);
}

template<typename MaskPixelT>
int afwImage::Mask<MaskPixelT>::addMaskPlane(const std::string& name)
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
        throw LSST_EXCEPT(pexExcept::RuntimeErrorException,
            (boost::format("Max number of planes (%1%) already used") % getNumPlanesMax()).str());
    }
}

/**
 * \brief set the name of a mask plane, with minimal checking.
 *
 * This is a private function and is mainly used by setMaskPlaneMetadata
 */
template<typename MaskPixelT>
int afwImage::Mask<MaskPixelT>::addMaskPlane(
    std::string name,   ///< new name of mask plane
    int planeId         ///< ID of mask plane to be (re)named
) {
    if (planeId < 0 || planeId >= getNumPlanesMax()) {
        throw LSST_EXCEPT(pexExcept::RangeErrorException,
            (boost::format("mask plane ID must be between 0 and %1%") % (getNumPlanesMax() - 1)).str());
    }

    _maskPlaneDict[name] = planeId;

    return planeId;
}

/**
 * \brief Clear all pixels of the specified mask and remove the plane from the mask plane dictionary
 *
 * Log a message if the mask plane name is invalid.
 */
template<typename MaskPixelT>
void afwImage::Mask<MaskPixelT>::removeMaskPlane(const std::string& name)
{
    int id;
    try {
        id = getMaskPlane(name);
        clearMaskPlane(id);
        _maskPlaneDict.erase(name);
        _myMaskDictVersion = ++_maskDictVersion;
        return;
    } catch (std::exception &e) {
        pexLog::Trace("afw.Mask", 0,
                      boost::format("%s Plane %s not present in this Mask") % e.what() % name);
        return;
    }
    
}

/**
 * \brief Return the bitmask corresponding to a plane ID, or 0 if invalid
 */
template<typename MaskPixelT>
MaskPixelT afwImage::Mask<MaskPixelT>::getBitMaskNoThrow(int planeId) {
    return (planeId >= 0 && planeId < getNumPlanesMax()) ? (1 << planeId) : 0;
}

/**
 * \brief Return the bitmask corresponding to plane ID
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if plane is invalid
 */
template<typename MaskPixelT>
MaskPixelT afwImage::Mask<MaskPixelT>::getBitMask(int planeId) {
    for (MaskPlaneDict::const_iterator i = _maskPlaneDict.begin(); i != _maskPlaneDict.end(); ++i) {
        if (planeId == i->second) {
            MaskPixelT const bitmask = getBitMaskNoThrow(planeId);
            if (bitmask == 0) {         // failed
                break;
            }
            return bitmask;
        }
    }
    throw LSST_EXCEPT(pexExcept::InvalidParameterException,
        (boost::format("Invalid mask plane ID: %d") % planeId).str());
}

/**
 * \brief Return the mask plane number corresponding to a plane name
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if plane is invalid
 */
template<typename MaskPixelT>
int afwImage::Mask<MaskPixelT>::getMaskPlane(const std::string& name) {
    int const plane = getMaskPlaneNoThrow(name);
    
    if (plane < 0) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,
            (boost::format("Invalid mask plane name: %s") % name).str());
    } else {
        return plane;
    }
}

/**
 * \brief Return the mask plane number corresponding to a plane name, or -1 if not found
 */
template<typename MaskPixelT>
int afwImage::Mask<MaskPixelT>::getMaskPlaneNoThrow(const std::string& name) {
    MaskPlaneDict::const_iterator plane = _maskPlaneDict.find(name);
    
    if (plane == _maskPlaneDict.end()) {
        return -1;
    } else {
        return plane->second;
    }
}

/**
 * \brief Return the bitmask corresponding to a plane name
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if plane is invalid
 */
template<typename MaskPixelT>
MaskPixelT afwImage::Mask<MaskPixelT>::getPlaneBitMask(const std::string& name) {
    return getBitMask(getMaskPlane(name));
}

/**
 * \brief Reset the maskPlane dictionary
 */
template<typename MaskPixelT>
void afwImage::Mask<MaskPixelT>::clearMaskPlaneDict() {
    _maskPlaneDict.clear();
    _myMaskDictVersion = ++_maskDictVersion;
}

/**
 * \brief Clear all the pixels
 */
template<typename MaskPixelT>
void afwImage::Mask<MaskPixelT>::clearAllMaskPlanes() {
    *this = 0;
}

/**
 * \brief Clear the specified bit in all pixels
 */
template<typename MaskPixelT>
void afwImage::Mask<MaskPixelT>::clearMaskPlane(int planeId) {
    *this &= ~getBitMask(planeId);
}

/**
 * \brief Adjust this mask to conform to the standard Mask class's mask plane dictionary,
 * adding any new mask planes to the standard.
 *
 * Ensures that this mask (presumably from some external source) has the same plane assignments
 * as the Mask class. If a change in plane assignments is needed, the bits within each pixel
 * are permuted as required.
 *
 * Any new mask planes found in this mask are added to unused slots in the Mask class's mask plane dictionary.
 */
template<typename MaskPixelT>
void afwImage::Mask<MaskPixelT>::conformMaskPlanes(
    MaskPlaneDict const &currentPlaneDict   ///< mask plane dictionary for this mask
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
    // We've made the planes match the current mask dictionary
    _myMaskDictVersion = _maskDictVersion;
}

/************************************************************************************************************/

/**
 * \brief get a reference to the specified pixel
 */
template<typename MaskPixelT>
typename afwImage::ImageBase<MaskPixelT>::PixelReference afwImage::Mask<MaskPixelT>::operator()(
    int x,  ///< x index
    int y   ///< y index
) {
    return this->ImageBase<MaskPixelT>::operator()(x, y);
}

/**
 * \brief get a reference to the specified pixel checking array bounds
 */
template<typename MaskPixelT>
typename afwImage::ImageBase<MaskPixelT>::PixelReference afwImage::Mask<MaskPixelT>::operator()(
    int x,                              ///< x index
    int y,                              ///< y index
    afwImage::CheckIndices const& check ///< Check array bounds?
) {
    return this->ImageBase<MaskPixelT>::operator()(x, y, check);
}

/**
 * \brief get the specified pixel (const version)
 */
template<typename MaskPixelT>
typename afwImage::ImageBase<MaskPixelT>::PixelConstReference afwImage::Mask<MaskPixelT>::operator()(
    int x,  ///< x index
    int y   ///< y index
) const {
    return this->ImageBase<MaskPixelT>::operator()(x, y);
}

/**
 * \brief get the specified pixel with array checking (const version)
 */
template<typename MaskPixelT>
typename afwImage::ImageBase<MaskPixelT>::PixelConstReference afwImage::Mask<MaskPixelT>::operator()(
    int x,                              ///< x index
    int y,                              ///< y index
    afwImage::CheckIndices const& check ///< Check array bounds?
) const {
    return this->ImageBase<MaskPixelT>::operator()(x, y, check);
}

/**
 * \brief is the specified mask plane set in the specified pixel?
 */
template<typename MaskPixelT>
bool afwImage::Mask<MaskPixelT>::operator()(
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
bool afwImage::Mask<MaskPixelT>::operator()(
    int x,                              ///< x index
    int y,                              ///< y index
    int planeId,                        ///< plane ID
    afwImage::CheckIndices const& check ///< Check array bounds?
) const {
    // !! converts an int to a bool
    return !!(this->ImageBase<MaskPixelT>::operator()(x, y, check) & getBitMask(planeId));
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
void afwImage::Mask<MaskPixelT>::operator|=(MaskPixelT const val) {
    transform_pixels(_getRawView(), _getRawView(), bl::ret<MaskPixelT>(bl::_1 | val));
}

/**
 * \brief OR a Mask into a Mask
 */
template<typename MaskPixelT>
void afwImage::Mask<MaskPixelT>::operator|=(Mask const &rhs) {
    checkMaskDictionaries(rhs);

    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("Images are of different size, %dx%d v %dx%d") %
                           this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()).str());
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), bl::ret<MaskPixelT>(bl::_1 | bl::_2));
}

/**
 * \brief AND a bitmask into a Mask
 */
template<typename MaskPixelT>
void afwImage::Mask<MaskPixelT>::operator&=(MaskPixelT const val) {
    transform_pixels(_getRawView(), _getRawView(), bl::ret<MaskPixelT>(bl::_1 & val));
}

/**
 * \brief AND a Mask into a Mask
 */
template<typename MaskPixelT>
void afwImage::Mask<MaskPixelT>::operator&=(Mask const &rhs) {
    checkMaskDictionaries(rhs);

    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("Images are of different size, %dx%d v %dx%d") %
                           this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()).str());
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), bl::ret<MaskPixelT>(bl::_1 & bl::_2));
}

/**
 * \brief XOR a bitmask into a Mask
 */
template<typename MaskPixelT>
void afwImage::Mask<MaskPixelT>::operator^=(MaskPixelT const val) {
    transform_pixels(_getRawView(), _getRawView(), bl::ret<MaskPixelT>(bl::_1 ^ val));
}

/**
 * \brief XOR a Mask into a Mask
 */
template<typename MaskPixelT>
void afwImage::Mask<MaskPixelT>::operator^=(Mask const &rhs) {
    checkMaskDictionaries(rhs);

    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("Images are of different size, %dx%d v %dx%d") %
                           this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()).str());
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), bl::ret<MaskPixelT>(bl::_1 ^ bl::_2));
}

/**
 * \brief Set the bit specified by "planeId" for pixels (x0, y) ... (x1, y)
 */
template<typename MaskPixelT>
void afwImage::Mask<MaskPixelT>::setMaskPlaneValues(int const planeId,
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
void afwImage::Mask<MaskPixelT>::addMaskPlanesToMetadata(lsst::daf::base::PropertySet::Ptr metadata) {
    if (!metadata) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "Null PropertySet::Ptr");
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
 * \brief Given a PropertySet that contains the MaskPlane assignments, setup the MaskPlanes.
 *
 * @returns a dictionary of mask plane name: plane ID
 */
template<typename MaskPixelT>
typename afwImage::Mask<MaskPixelT>::MaskPlaneDict afwImage::Mask<MaskPixelT>::parseMaskPlaneMetadata(
        lsst::daf::base::PropertySet::Ptr const metadata ///< metadata from a Mask
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
                                      (boost::format("File specifies plane %s has same value (%d) as %s") %
                                       planeName % planeId % j->first).str());
                }
            }
            // build new entry
            if (numPlanesUsed >= getNumPlanesMax()) {
                // Max number of planes already allocated
                throw LSST_EXCEPT(pexExcept::RuntimeErrorException,
                    (boost::format("Max number of planes (%1%) already used") % getNumPlanesMax()).str());
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
void afwImage::Mask<MaskPixelT>::printMaskPlanes() {
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
static typename afwImage::Mask<MaskPixelT>::MaskPlaneDict initMaskPlanes() {
    typename afwImage::Mask<MaskPixelT>::MaskPlaneDict planeDict =
        typename afwImage::Mask<MaskPixelT>::MaskPlaneDict();

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
std::string const afwImage::Mask<MaskPixelT>::maskPlanePrefix("MP_");

template<typename MaskPixelT>
typename afwImage::Mask<MaskPixelT>::MaskPlaneDict afwImage::Mask<MaskPixelT>::_maskPlaneDict =
                                                          initMaskPlanes<MaskPixelT>();

template<typename MaskPixelT>
int afwImage::Mask<MaskPixelT>::_maskDictVersion = 0;    // version number for bitplane dictionary

//
// Explicit instantiations
//
template class afwImage::Mask<afwImage::MaskPixel>;
