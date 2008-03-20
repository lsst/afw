// -*- lsst-c++ -*-
// Implementations of Mask class methods

#include "lsst/fw/Mask.h"

using namespace lsst::fw;

template<typename MaskPixelT>
Mask<MaskPixelT>::Mask(MaskPlaneDict const& planeDefs) :
    lsst::mwi::data::LsstBase(typeid(this)),
    _vwImagePtr(new vw::ImageView<MaskPixelT>()),
    _metaData(lsst::mwi::data::SupportFactory::createPropertyNode("FitsMetaData")),
    _offsetRows(0), _offsetCols(0),
    _MyMaskDictVersion(_MaskDictVersion) {

    lsst::mwi::utils::Trace("fw.Mask", 5,
              boost::format("Number of mask planes: %d") % getNumPlanesMax());

    if (planeDefs.size() > 0) {
        _maskPlaneDict = planeDefs;
        _MyMaskDictVersion = ++_MaskDictVersion;
    }
}

template<typename MaskPixelT>
Mask<MaskPixelT>::Mask(MaskIVwPtrT vwImagePtr, MaskPlaneDict const& planeDefs): 
    lsst::mwi::data::LsstBase(typeid(this)),
    _vwImagePtr(vwImagePtr),
    _metaData(lsst::mwi::data::SupportFactory::createPropertyNode("FitsMetaData")),
    _offsetRows(0), _offsetCols(0),
    _MyMaskDictVersion(_MaskDictVersion) {
    
    lsst::mwi::utils::Trace("fw.Mask", 5,
              boost::format("Number of mask planes: %d") % getNumPlanesMax());

    if (planeDefs.size() > 0) {
        _maskPlaneDict = planeDefs;
        _MyMaskDictVersion = ++_MaskDictVersion;
    }
}

template<typename MaskPixelT>
Mask<MaskPixelT>::Mask(int ncols, int nrows, MaskPlaneDict const& planeDefs) :
    lsst::mwi::data::LsstBase(typeid(this)),
    _vwImagePtr(new vw::ImageView<MaskPixelT>(ncols, nrows)),
    _metaData(lsst::mwi::data::SupportFactory::createPropertyNode("FitsMetaData")),
    _offsetRows(0), _offsetCols(0),
    _MyMaskDictVersion(_MaskDictVersion) {

    lsst::mwi::utils::Trace("fw.Mask", 5,
              boost::format("Number of mask planes: %d") % getNumPlanesMax());

    if (planeDefs.size() > 0) {
        _maskPlaneDict = planeDefs;
        _MyMaskDictVersion = ++_MaskDictVersion;
    }
}

/*
 * Avoiding the default assignment operator seems to solve ticket 144 (memory leaks in the Python code),
 * but I don't know why. Explicitly resetting the shared pointers before setting them is not the answer
 * (because commenting it out makes no difference) and the rest of the code should be identical
 * to the default assignment operator.
 */
template<typename MaskPixelT>
Mask<MaskPixelT>& Mask<MaskPixelT>::operator= (const Mask<MaskPixelT>& rhs) {
    if (&rhs != this) {   // beware of self assignment: mask = mask;
        _vwImagePtr.reset();
        _vwImagePtr = rhs._vwImagePtr;
        _maskPlaneDict = rhs._maskPlaneDict;
        _metaData = rhs._metaData;
        _offsetRows = rhs._offsetRows;
        _offsetCols = rhs._offsetCols;
    }
    
    return *this;
}

template<typename MaskPixelT>
lsst::mwi::data::DataProperty::PtrType Mask<MaskPixelT>::getMetaData()
{
    return _metaData;
}

template<typename MaskPixelT>
void Mask<MaskPixelT>::readFits(const std::string& fileName, bool conformMasks, int hdu)
{
    LSSTFitsResource<MaskPixelT> fitsRes;
    fitsRes.readFits(fileName, *_vwImagePtr, _metaData, hdu);
    if (conformMasks==true) {
        MaskPlaneDict masterPlaneDefs = _maskPlaneDict;
        parseMaskPlaneMetaData(_metaData);
        conformMaskPlanes(masterPlaneDefs);
    } else {
        parseMaskPlaneMetaData(_metaData);
    }        
}

template<typename MaskPixelT>
void Mask<MaskPixelT>::writeFits(const std::string& fileName)
{
    LSSTFitsResource<MaskPixelT> fitsRes;
    addMaskPlaneMetaData(_metaData);
    fitsRes.writeFits(*_vwImagePtr, _metaData, fileName);
}

template<typename MaskPixelT>
int Mask<MaskPixelT>::addMaskPlane(const std::string& name)
{
    int const id = getMaskPlane(name);

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
        throw OutOfPlaneSpace("Max number of planes already used")
            << lsst::mwi::data::DataProperty("numPlanesUsed", _maskPlaneDict.size())
            << lsst::mwi::data::DataProperty("numPlanesMax", getNumPlanesMax());
    }
}

// This is a private function.  It sets the plane of the given planeId to be name
// with minimal checking.   Mainly used by setMaskPlaneMetadata

template<typename MaskPixelT> 
int Mask<MaskPixelT>::addMaskPlane(std::string name, int planeId)
{
    if (planeId < 0 || planeId >= getNumPlanesMax()) {
        throw;
    }

    _maskPlaneDict[name] = planeId;

    return planeId;
}

template<typename MaskPixelT>
void Mask<MaskPixelT>::removeMaskPlane(const std::string& name)
{
     int id;
     try {
        id = getMaskPlane(name);
        clearMaskPlane(id);
        _maskPlaneDict.erase(name);
        _MyMaskDictVersion = ++_MaskDictVersion;
        return;
     } catch (std::exception &e) {
        lsst::mwi::utils::Trace("fw.Mask", 0,
                   boost::format("%s Plane %s not present in this Mask") % e.what() % name);
        return;
     }
     
}

template<typename MaskPixelT>
void Mask<MaskPixelT>::getMaskPlane(const std::string& name,
                                              int& plane) const {
    plane = getMaskPlane(name);
    if (plane < 0) {
        throw NoMaskPlane("Failed to find maskPlane " + name);
    }
}

// \brief Return the bitmask corresponding to plane
//
// @throw lsst::mwi::exceptions::InvalidParameter if plane is invalid
template<typename MaskPixelT>
typename Mask<MaskPixelT>::MaskChannelT Mask<MaskPixelT>::getBitMask(int plane) const {
    for (typename MaskPlaneDict::const_iterator i = _maskPlaneDict.begin(); i != _maskPlaneDict.end(); ++i) {
        if (plane == i->second) {
            return (1 << plane);
        }
    }

    throw lsst::mwi::exceptions::InvalidParameter(boost::format("Invalid mask plane: %d") % plane);
}


template<typename MaskPixelT>
int Mask<MaskPixelT>::getMaskPlane(const std::string& name) const {

    typename Mask<MaskPixelT>::MaskPlaneDict::const_iterator plane = _maskPlaneDict.find(name);
    
    if (plane == _maskPlaneDict.end()) {
        return -1;
    } else {
        return plane->second;
    }
}

template<typename MaskPixelT>
bool Mask<MaskPixelT>::getPlaneBitMask(const std::string& name,
                                                 MaskChannelT& bitMask) const {
    int plane = getMaskPlane(name);
    if (plane < 0) {
        lsst::mwi::utils::Trace("fw.Mask", 1, boost::format("Plane %s not present in this Mask") % name);
        return false;
    }

    bitMask = getBitMask(plane);
    return true;
}

template<typename MaskPixelT>
typename Mask<MaskPixelT>::MaskChannelT Mask<MaskPixelT>::getPlaneBitMask(
    const std::string& name
) const {
    return getBitMask(getMaskPlane(name));
}

template<typename MaskPixelT>
void Mask<MaskPixelT>::clearAllMaskPlanes() {
    _maskPlaneDict.clear();
    _MyMaskDictVersion = ++_MaskDictVersion;

    for (unsigned int y = 0; y != getRows(); y++) {
        for (unsigned int x = 0; x != getCols(); x++) {
            (*_vwImagePtr)(x,y) = 0;
        }
     }
}

// clearMaskPlane(int plane) clears the bit specified by "plane" in all pixels in the mask
//
template<typename MaskPixelT>
void Mask<MaskPixelT>::clearMaskPlane(int plane) {
    MaskPixelT const bitMask = getBitMask(plane);

    for (unsigned int y = 0; y < getRows(); y++) {
        for (unsigned int x = 0; x < getCols(); x++) {
            (*_vwImagePtr)(x,y) &= ~bitMask;
        }
     }
}

// \brief Convert the current maskPlaneDict to the canonical one defined in Mask
//
// conformMaskPlanes ensures that this Mask (presumably from some external source)
// has the same plane assignments as Mask.  If a change in plane assignments is needed,
// the bits within each pixel are permuted as required
//
template<typename MaskPixelT>
void Mask<MaskPixelT>::conformMaskPlanes(MaskPlaneDict currentPlaneDict) {

    if (_maskPlaneDict == currentPlaneDict) {
        _MyMaskDictVersion = _MaskDictVersion;
        return;   // nothing to do
    }
    //
    // Find out which planes need to be permuted
    //
    MaskChannelT keepBitmask = 0;       // mask of bits to keep
    MaskChannelT canonicalMask[sizeof(MaskChannelT)*8]; // bits in lsst::fw::Mask that should be
    MaskChannelT currentMask[sizeof(MaskChannelT)*8]; //           mapped to these bits
    int numReMap = 0;

    for (MaskPlaneDict::const_iterator i = currentPlaneDict.begin(); i != currentPlaneDict.end() ; i++) {
        std::string const name = i->first; // name of mask plane
        int const currentPlaneNumber = i->second; // plane number currently in use
        int canonicalPlaneNumber = getMaskPlane(name); // plane number in lsst::fw::Mask

        if (canonicalPlaneNumber < 0) {                  // no such plane; add it
            canonicalPlaneNumber = addMaskPlane(name);
        }
        
        if (canonicalPlaneNumber == currentPlaneNumber) {
            keepBitmask |= getBitMask(canonicalPlaneNumber); // bit is unchanged, so preserve it
        } else {
            canonicalMask[numReMap] = getBitMask(canonicalPlaneNumber);
            currentMask[numReMap]   = getBitMask(currentPlaneNumber);
            numReMap++;
        }
    }

    // Now loop over all pixels in Mask
    if (numReMap > 0) {
        for (unsigned int y = 0; y < getRows(); y++) {
            for (unsigned int x = 0; x < getCols(); x++) {
                MaskChannelT const pixel = (*_vwImagePtr)(x,y); // current value
                MaskChannelT newPixel = pixel & keepBitmask; // value of invariant mask bits
                for (int j = 0; j < numReMap; j++) {
                    if (pixel & currentMask[j]) newPixel |= canonicalMask[j];
                }
                (*_vwImagePtr)(x,y) = newPixel;
            }
        }
    }
    // We've made the planes match the current mask dictionary
    _MyMaskDictVersion = _MaskDictVersion;
}


/**
 * \brief Set the bit specified by "plane" for each pixel in the pixelList
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::setMaskPlaneValues(int plane, std::list<PixelCoord> pixelList) {
    MaskPixelT const bitMask = getBitMask(plane);

    for (std::list<PixelCoord>::iterator i = pixelList.begin(); i != pixelList.end(); i++) {
        PixelCoord coord = *i;
        (*_vwImagePtr)(coord.x, coord.y) |= bitMask;
    }
}

/**
 * \brief Set the bit specified by "plane" for pixels (x0, y) ... (x1, y)
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::setMaskPlaneValues(const int plane, const int x0, const int x1, const int y) {
    MaskPixelT const bitMask = getBitMask(plane);
    
    for (int x = x0; x <= x1; x++) {
        (*_vwImagePtr)(x, y) |= bitMask;
    }
}

/**
 * \brief Set the bit specified by "plane" for each pixel for which selectionFunc(pixel) returns true
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::setMaskPlaneValues(int plane, MaskPixelBooleanFunc<MaskPixelT> selectionFunc) {
    MaskPixelT const bitMask = getBitMask(plane);

    for (unsigned int y = 0; y < getRows(); y++) {
        for (unsigned int x = 0; x < getCols(); x++) {
            if (selectionFunc((*_vwImagePtr)(x,y)) == true) {
                (*_vwImagePtr)(x,y) |= bitMask;
            }
          }
    }
}

/**
 * \brief Return the number of pixels within maskRegion for which testFunc(pixel) returns true
 *
 * PROBABLY WANT maskRegion to default to whole Mask
 */
template<typename MaskPixelT>
int Mask<MaskPixelT>::countMask(
    MaskPixelBooleanFunc<MaskPixelT>& testFunc,
    const vw::BBox2i maskRegion
) const {
    int count = 0;
    Vector<int32, 2> ulCorner = maskRegion.min();
    Vector<int32, 2> lrCorner = maskRegion.max();

    for (int y=ulCorner[1]; y<lrCorner[1]; y++) {
        for (int x=ulCorner[0]; x<lrCorner[0]; x++) {
            if (testFunc((*_vwImagePtr)(x,y)) == true) {
                count += 1;
            }
          }
    }
     return count;
}

template<typename MaskPixelT>
typename Mask<MaskPixelT>::MaskPtrT Mask<MaskPixelT>::getSubMask(const vw::BBox2i maskRegion) const {

    // Check that maskRegion is completely inside the mask
    
    vw::BBox2i maskBoundary(0, 0, getCols(), getRows());
    if (!maskBoundary.contains(maskRegion)) {
        throw lsst::mwi::exceptions::InvalidParameter(boost::format("getSubMask region not contained within Mask"));
    }

    MaskIVwPtrT croppedMask(new MaskIVwT());
    *croppedMask = copy(crop(*_vwImagePtr, maskRegion));
    MaskPtrT newMask(new Mask<MaskPixelT>(croppedMask));
    Vector<int, 2> bboxOffset = maskRegion.min();
    newMask->setOffsetRows(bboxOffset[1] + _offsetRows);
    newMask->setOffsetCols(bboxOffset[0] + _offsetCols);

    // Now copy Maskplane info:
    newMask->_maskPlaneDict = _maskPlaneDict;
    newMask->_metaData = _metaData;

    return newMask;
}

/**
 * \brief Given a Mask, insertMask, place it into this Mask as directed by maskRegion.
 *
 * \throw lsst::mwi::exceptions::Exception if maskRegion is not of the same size as insertMask.
 *
 * Maybe generate an exception if offsets are not consistent?
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::replaceSubMask(const vw::BBox2i maskRegion, MaskPtrT insertMask)
{
    try {
        crop(*_vwImagePtr, maskRegion) = *(insertMask->_vwImagePtr);
    } catch (std::exception eex) {
        throw lsst::mwi::exceptions::Runtime(std::string("in ") + __func__);
    } 
}

template<typename MaskPixelT>
typename Mask<MaskPixelT>::MaskChannelT Mask<MaskPixelT>::operator ()(int x, int y) const
{
//      std::cout << x << " " << y << " " << (void *)(*_vwImagePtr)(x, y) << std::endl;
     return (*_vwImagePtr)(x, y);
}

template<typename MaskPixelT>
bool Mask<MaskPixelT>::operator ()(int x, int y, int plane) const
{
     return ((*_vwImagePtr)(x, y) & getBitMask(plane)) != 0;
}

template<typename MaskPixelT>
bool MaskPixelBooleanFunc<MaskPixelT>::operator() (MaskPixelT) const {
    throw lsst::mwi::exceptions::Runtime(boost::format("You can't get here: %s:%d") % __FILE__ % __LINE__);
    return true;
}

template<typename MaskPixelT>
Mask<MaskPixelT>& Mask<MaskPixelT>::operator |= (const Mask<MaskPixelT>& inputMask)
{
    // Need to check for identical sizes, and presence of all needed planes
    if (getCols() != inputMask.getCols() || getRows() != inputMask.getRows()) {
        throw lsst::mwi::exceptions::Runtime("Sizes do not match");
    }

    checkMaskDictionaries(inputMask);
     
    // Now, can iterate through the MaskImages, or'ing the input pixels into this MaskImage

    for (unsigned int y = 0; y < getRows(); y++) {
        for (unsigned int x = 0; x < getCols(); x++) {
            (*_vwImagePtr)(x,y) |= inputMask(x,y);
        }
    }

    return *this;
}

/**
 * \brief OR a bitmask into a Mask
 *
 * \return Modified Mask
 */
template<typename MaskPixelT>
Mask<MaskPixelT>& Mask<MaskPixelT>::operator |= (MaskPixelT const inputMask)
{
    for (unsigned int y = 0; y < getRows(); y++) {
        for (unsigned int x = 0; x < getCols(); x++) {
            (*_vwImagePtr)(x,y) |= inputMask;
        }
    }

    return *this;
}

/**
 * \brief AND a bitmask into a Mask
 *
 * \return Modified Mask
 */
template<typename MaskPixelT>
Mask<MaskPixelT>& Mask<MaskPixelT>::operator &= (MaskPixelT const inputMask)
{
    for (unsigned int y = 0; y < getRows(); y++) {
        for (unsigned int x = 0; x < getCols(); x++) {
            (*_vwImagePtr)(x,y) &= inputMask;
        }
    }

    return *this;
}

/**
 * \brief Given a DataProperty, replace any existing MaskPlane assignments with the current ones.
 *
 * \throw Throws lsst::mwi::exceptions::InvalidParameter if given DataProperty is not a node
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::addMaskPlaneMetaData(lsst::mwi::data::DataProperty::PtrType rootPtr) {
     if( rootPtr->isNode() != true ) {
        throw lsst::mwi::exceptions::InvalidParameter( "Given DataProperty object is not a node" );
        
     }

    // First, clear existing MaskPlane metadata
    rootPtr->deleteAll( maskPlanePrefix +".*", false );

    // Add new MaskPlane metadata
    for (MaskPlaneDict::iterator i = _maskPlaneDict.begin(); i != _maskPlaneDict.end() ; ++i) {
        std::string const planeName = i->first;
        int const planeNumber = i->second;
        
        if (planeName != "") {
            rootPtr->addProperty(
                lsst::mwi::data::DataProperty::PtrType(new lsst::mwi::data::DataProperty(Mask::maskPlanePrefix + planeName, planeNumber)));
        }
    }
}


/**
 * \brief Given a DataProperty that contains the MaskPlane assignments setup the MaskPlanes.
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::parseMaskPlaneMetaData(const lsst::mwi::data::DataProperty::PtrType rootPtr) {

    lsst::mwi::data::DataProperty::iteratorRangeType range = rootPtr->searchAll( maskPlanePrefix +".*" );
    if (std::distance(range.first, range.second) == 0) {
        return;
    }

    // Clear all existing MaskPlanes

    clearAllMaskPlanes();

    // Iterate through matching keyWords
    lsst::mwi::data::DataProperty::ContainerIteratorType iter;
    for( iter = range.first; iter != range.second; iter++ ) {
        lsst::mwi::data::DataProperty::PtrType dpPtr = *iter;
        // split off the "MP_" to get the planeName
        std::string keyWord = dpPtr->getName();
        std::string planeName = keyWord.substr(maskPlanePrefix.size());
        // will throw an exception if the found item does not contain const int
        int planeId = boost::any_cast<const int>(dpPtr->getValue());
        addMaskPlane(planeName, planeId);
    }

}

template<typename MaskPixelT>
void Mask<MaskPixelT>::printMaskPlanes() const {
    for (MaskPlaneDict::const_iterator i = _maskPlaneDict.begin(); i != _maskPlaneDict.end() ; i++) {
        std::string const planeName = i->first;
        int const planeNumber = i->second;

        std::cout << "Plane " << planeNumber << " -> " << planeName << std::endl;
    }
}

template<typename MaskPixelT>
typename Mask<MaskPixelT>::MaskPlaneDict Mask<MaskPixelT>::getMaskPlaneDict() const {
    return _maskPlaneDict;
}

template<typename MaskPixelT>
void Mask<MaskPixelT>::setOffsetRows(unsigned int offset) {
    _offsetRows = offset;
}

template<typename MaskPixelT>
void Mask<MaskPixelT>::setOffsetCols(unsigned int offset)
{
    _offsetCols = offset;
}

/*
 * Default Mask planes
 */
template<typename MaskPixelT> 
static typename Mask<MaskPixelT>::MaskPlaneDict initMaskPlanes() {

    typename Mask<MaskPixelT>::MaskPlaneDict planeDict = typename Mask<MaskPixelT>::MaskPlaneDict();

    int i = -1;
    planeDict["BAD"] = ++i;
    planeDict["CR"] = ++i;
    planeDict["EDGE"] = ++i;
    planeDict["INTRP"] = ++i;
    planeDict["SAT"] = ++i;

    return planeDict;
}

/*
 * Static members of Mask
 */
template<typename MaskPixelT> 
std::string const Mask<MaskPixelT>::maskPlanePrefix("MP_");

template<typename MaskPixelT> 
typename Mask<MaskPixelT>::MaskPlaneDict Mask<MaskPixelT>::_maskPlaneDict = initMaskPlanes<MaskPixelT>();

template<typename MaskPixelT> 
int Mask<MaskPixelT>::_MaskDictVersion = 0;    // version number for bitplane dictionary

/************************************************************************************************************/
//
// Explicit instantiations
//
template class Mask<unsigned char>;
template class MaskPixelBooleanFunc<unsigned char>;

template class Mask<maskPixelType>;
template class MaskPixelBooleanFunc<maskPixelType>;
