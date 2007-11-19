// -*- lsst-c++ -*-
// Implementations of Mask class methods
// This file can NOT be separately compiled!   It is included by Mask.h

template<typename MaskPixelT>
lsst::fw::Mask<MaskPixelT>::Mask(MaskPlaneDict planeDefs) :
    lsst::mwi::data::LsstBase(typeid(this)),
    _vwImagePtr(new vw::ImageView<MaskPixelT>()),
    _metaData(lsst::mwi::data::SupportFactory::createPropertyNode("FitsMetaData")) {

    lsst::mwi::utils::Trace("fw.Mask", 5,
              boost::format("Number of mask planes: %d") % getNumPlanesMax());

    // Check whether planeDefs is coming in as the default empty map

    _numPlanesUsed = 0;

    if (planeDefs.size() == 0) {
        for (int i=0; i<getNumPlanesMax(); i++) {
            _maskPlaneDict[i] = "";
        }

    } else {
        _maskPlaneDict = planeDefs;
        if (_maskPlaneDict.size() != static_cast<unsigned int>(getNumPlanesMax())) {
            // raise some exception
        }
        
        for (int i=0; i<getNumPlanesMax(); i++) {
            if (_maskPlaneDict[i] != "") {
                _numPlanesUsed++;
            }
        }
    }

     _offsetRows = 0;
     _offsetCols = 0;

}

template<typename MaskPixelT>
lsst::fw::Mask<MaskPixelT>::Mask(MaskIVwPtrT vwImagePtr, MaskPlaneDict planeDefs): 
    lsst::mwi::data::LsstBase(typeid(this)),
    _vwImagePtr(vwImagePtr),
    _metaData(lsst::mwi::data::SupportFactory::createPropertyNode("FitsMetaData")) {

    lsst::mwi::utils::Trace("fw.Mask", 5,
              boost::format("Number of mask planes: %d") % getNumPlanesMax());

    // Check whether planeDefs is coming in as the default empty map

    _numPlanesUsed = 0;

    if (planeDefs.size() == 0) {
        for (int i=0; i<getNumPlanesMax(); i++) {
            _maskPlaneDict[i] = "";
        }

    } else {
        _maskPlaneDict = planeDefs;
        if (_maskPlaneDict.size() != static_cast<unsigned int>(getNumPlanesMax())) {
            // raise some exception
        }
        
        for (int i=0; i<getNumPlanesMax(); i++) {
            if (_maskPlaneDict[i] != "") {
                _numPlanesUsed++;
            }
        }
    }

     _offsetRows = 0;
     _offsetCols = 0;
}

template<typename MaskPixelT>
lsst::fw::Mask<MaskPixelT>::Mask(int ncols, int nrows, MaskPlaneDict planeDefs) :
    lsst::mwi::data::LsstBase(typeid(this)),
    _vwImagePtr(new vw::ImageView<MaskPixelT>(ncols, nrows)),
    _metaData(lsst::mwi::data::SupportFactory::createPropertyNode("FitsMetaData")) {

    lsst::mwi::utils::Trace("fw.Mask", 5,
              boost::format("Number of mask planes: %d") % getNumPlanesMax());

    // Check whether planeDefs is coming in as the default empty map

    _numPlanesUsed = 0;

    if (planeDefs.size() == 0) {
        for (int i=0; i<getNumPlanesMax(); i++) {
            _maskPlaneDict[i] = "";
        }

    } else {
        _maskPlaneDict = planeDefs;
        if (_maskPlaneDict.size() != static_cast<unsigned int>(getNumPlanesMax())) {
            // raise some exception
        }
        
        for (int i=0; i<getNumPlanesMax(); i++) {
            if (_maskPlaneDict[i] != "") {
                _numPlanesUsed++;
            }
        }
    }

     _offsetRows = 0;
     _offsetCols = 0;
}

/*
 * Avoiding the default assignment operator seems to solve ticket 144 (memory leaks in the Python code),
 * but I don't know why. Explicitly resetting the shared pointers before setting them is not the answer
 * (because commenting it out makes no difference) and the rest of the code should be identical
 * to the default assignment operator.
 */
template<typename MaskPixelT>
lsst::fw::Mask<MaskPixelT>& lsst::fw::Mask<MaskPixelT>::operator= (const Mask<MaskPixelT>& rhs) {
    if (&rhs != this) {   // beware of self assignment: mask = mask;
        _vwImagePtr.reset();
        _vwImagePtr = rhs._vwImagePtr;
        _maskPlaneDict = rhs._maskPlaneDict;
        _numPlanesUsed = rhs._numPlanesUsed;
        _metaData = rhs._metaData;
        _offsetRows = rhs._offsetRows;
        _offsetCols = rhs._offsetCols;
    }
    
    return *this;
}

template<typename MaskPixelT>
lsst::mwi::data::DataProperty::PtrType lsst::fw::Mask<MaskPixelT>::getMetaData()
{
    return _metaData;
}

template<typename MaskPixelT>
void lsst::fw::Mask<MaskPixelT>::readFits(const std::string& fileName, bool conformMasks, int hdu)
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
void lsst::fw::Mask<MaskPixelT>::writeFits(const std::string& fileName)
{
    LSSTFitsResource<MaskPixelT> fitsRes;
    addMaskPlaneMetaData(_metaData);
    fitsRes.writeFits(*_vwImagePtr, _metaData, fileName);
}

template<typename MaskPixelT>
int lsst::fw::Mask<MaskPixelT>::addMaskPlane(const std::string& name)
{
     int id;
     try {
        getMaskPlane(name, id);
     return id;
     }
     catch(lsst::fw::NoMaskPlane &e) {
        // build new entry
        if (_numPlanesUsed < getNumPlanesMax()) {
            // find first entry in dict with null name
            for (std::map<int, std::string>::iterator i=_maskPlaneDict.begin(); i != _maskPlaneDict.end(); ++i) {
                if (i->second == "") {
                    i->second = name;
                    _numPlanesUsed++; 
                    return i->first;
                }
            }
            // No free space found for new plane addition
            throw lsst::fw::OutOfPlaneSpace("No space to add new plane")
                << lsst::mwi::data::DataProperty("numPlanesUsed", _numPlanesUsed) 
                << lsst::mwi::data::DataProperty("numPlanesMax", getNumPlanesMax());
        } else {
            // Max number of planes already allocated
          throw lsst::fw::OutOfPlaneSpace("Max number of planes already used")
              << lsst::mwi::data::DataProperty("numPlanesUsed", _numPlanesUsed)
              << lsst::mwi::data::DataProperty("numPlanesMax", getNumPlanesMax());
        }
    }
}

// This is a private function.  It sets the plane of the given planeId to be name
// with minimal checking.   Mainly used by setMaskPlaneMetadata

template<typename MaskPixelT> 
int lsst::fw::Mask<MaskPixelT>::addMaskPlane(std::string name, int planeId)
{
    if (planeId <0 || planeId >= getNumPlanesMax()) {
        throw;
    }

    std::string oldName = _maskPlaneDict[planeId];
    _maskPlaneDict[planeId] = name;
    if (oldName == "") _numPlanesUsed++;
    return planeId;
}

template<typename MaskPixelT>
void lsst::fw::Mask<MaskPixelT>::removeMaskPlane(const std::string& name)
{
     int id;
     try {
        getMaskPlane(name, id);
        clearMaskPlane(id);
        _maskPlaneDict[id] = "";
        _numPlanesUsed--;
        return;
     }
     catch (std::exception &e) {
        lsst::mwi::utils::Trace("fw.Mask", 0,
                   boost::format("%s Plane %s not present in this Mask") % e.what() % name);
        return;
     }
     
}

template<typename MaskPixelT>
void lsst::fw::Mask<MaskPixelT>::getMaskPlane(const std::string& name,
                                    int& plane) const {
     for(std::map<int, std::string>::const_iterator i=_maskPlaneDict.begin(); i != _maskPlaneDict.end(); ++i) {
        if (i->second == name) {
            plane = i->first;
            return ;
        }
     }
     plane = -1;
     throw lsst::fw::NoMaskPlane("Failed to find maskPlane " + name);
}

template<typename MaskPixelT>
int lsst::fw::Mask<MaskPixelT>::getMaskPlane(const std::string& name) const {
    for(std::map<int, std::string>::const_iterator i=_maskPlaneDict.begin(); i != _maskPlaneDict.end(); ++i) {
        if (i->second == name) {
            return i->first;
        }
    }
    return -1;
}

template<typename MaskPixelT>
bool lsst::fw::Mask<MaskPixelT>::getPlaneBitMask(const std::string& name,
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
typename lsst::fw::Mask<MaskPixelT>::MaskChannelT lsst::fw::Mask<MaskPixelT>::getPlaneBitMask(
    const std::string& name
) const {
    return getBitMask(getMaskPlane(name));
}

template<typename MaskPixelT>
void lsst::fw::Mask<MaskPixelT>::clearAllMaskPlanes() {
     for (int i=0; i<getNumPlanesMax(); i++) {
        _maskPlaneDict[i] = "";
     }
    _numPlanesUsed = 0;
}

// clearMaskPlane(int plane) clears the bit specified by "plane" in all pixels in the mask
//
template<typename MaskPixelT>
void lsst::fw::Mask<MaskPixelT>::clearMaskPlane(int plane) {
    for (unsigned int y = 0; y < getRows(); y++) {
        for (unsigned int x = 0; x < getCols(); x++) {
           (*_vwImagePtr)(x,y) = (*_vwImagePtr)(x,y) & ~getBitMask(plane);
        }
     }
}

// conformMaskPlanes ensures that this Mask has the same plane assignments as
// masterMask.   If a change in plane assignments is needed, the bits within 
// each pixel are permuted as required
//
template<typename MaskPixelT>
void lsst::fw::Mask<MaskPixelT>::conformMaskPlanes(MaskPlaneDict masterPlaneDict) {

    if (_maskPlaneDict == masterPlaneDict)
        return;   // nothing to do

    MaskChannelT bitMasks[sizeof(MaskChannelT)*8];
    MaskChannelT myMask[sizeof(MaskChannelT)*8];
    MaskChannelT masterMask[sizeof(MaskChannelT)*8];
    int numReMap = 0;

    for (int i=0; i<getNumPlanesMax(); i++) {
        bitMasks[i] = getBitMask(i);
    }

    int masterNumPlanesUsed = 0;
    for (int i=0; i<getNumPlanesMax(); i++) {
        std::string masterId = masterPlaneDict[i];
        // determine i -> myJ
        // lookup masterId in _maskPlaneDict -> myJ; myJ=-1 if masterId=="" or masterId not found in _maskPlaneDict
        if (masterId != "") {
            ++masterNumPlanesUsed;
            int myJ = getMaskPlane(masterId); 
            if (myJ != -1 && myJ != i) {
                myMask[numReMap] = bitMasks[myJ];
                masterMask[numReMap] = bitMasks[i];
                numReMap++;
            }
        }
    }

    // warning if no corresponding plane in master.   This one will be dropped

    // Now loop over all pixels in Mask
    if (numReMap > 0) {
        for (unsigned int y = 0; y < getRows(); y++) {
            for (unsigned int x = 0; x < getCols(); x++) {
                MaskChannelT newPixel  = 0;
                MaskChannelT pixel = (*_vwImagePtr)(x,y);
                for (int j=0; j<numReMap; j++) {
                    if (pixel && myMask[j]) newPixel |= masterMask[j];
                }
                (*_vwImagePtr)(x,y) = newPixel;
            }
        }
    }

    _maskPlaneDict = masterPlaneDict;
    _numPlanesUsed = masterNumPlanesUsed;
}


/**
 * \brief Set the bit specified by "plane" for each pixel in the pixelList
 */
template<typename MaskPixelT>
void lsst::fw::Mask<MaskPixelT>::setMaskPlaneValues(int plane, std::list<PixelCoord> pixelList) {
//    std::cout << "setMaskPlaneValues " << pixelList.size() << std::endl;
    for (std::list<PixelCoord>::iterator i = pixelList.begin(); i != pixelList.end(); i++) {
        PixelCoord coord = *i;
        (*_vwImagePtr)(coord.x, coord.y) = (*_vwImagePtr)(coord.x, coord.y) | getBitMask(plane);
//        std::cout << "Set: " << coord.x << " " << coord.y << " " << (void *)getBitMask(plane) << " " << (*_vwImagePtr)(coord.x, coord.y) << std::endl;
    }
}

/**
 * \brief Set the bit specified by "plane" for pixels (x0, y) ... (x1, y)
 */
template<typename MaskPixelT>
void lsst::fw::Mask<MaskPixelT>::setMaskPlaneValues(const int plane, const int x0, const int x1, const int y) {
    for (int x = x0; x <= x1; x++) {
        (*_vwImagePtr)(x, y) |= getBitMask(plane);
    }
}

/**
 * \brief Set the bit specified by "plane" for each pixel for which selectionFunc(pixel) returns true
 */
template<typename MaskPixelT>
void lsst::fw::Mask<MaskPixelT>::setMaskPlaneValues(int plane, MaskPixelBooleanFunc<MaskPixelT> selectionFunc) {
    //  Should check plane for legality here...

    for (unsigned int y = 0; y < getRows(); y++) {
        for (unsigned int x = 0; x < getCols(); x++) {
            if (selectionFunc((*_vwImagePtr)(x,y)) == true) {
                (*_vwImagePtr)(x,y) = (*_vwImagePtr)(x,y) | getBitMask(plane);
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
int lsst::fw::Mask<MaskPixelT>::countMask(
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
typename lsst::fw::Mask<MaskPixelT>::MaskPtrT lsst::fw::Mask<MaskPixelT>::getSubMask(const vw::BBox2i maskRegion) const {

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
    newMask->_numPlanesUsed = _numPlanesUsed;
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
void lsst::fw::Mask<MaskPixelT>::replaceSubMask(const vw::BBox2i maskRegion, MaskPtrT insertMask)
{
    try {
        crop(*_vwImagePtr, maskRegion) = *(insertMask->_vwImagePtr);
    } catch (std::exception eex) {
        throw lsst::mwi::exceptions::Runtime(std::string("in ") + __func__);
    } 
}

template<typename MaskPixelT>
typename lsst::fw::Mask<MaskPixelT>::MaskChannelT lsst::fw::Mask<MaskPixelT>::operator ()(int x, int y) const
{
//      std::cout << x << " " << y << " " << (void *)(*_vwImagePtr)(x, y) << std::endl;
     return (*_vwImagePtr)(x, y);
}

template<typename MaskPixelT>
bool lsst::fw::Mask<MaskPixelT>::operator ()(int x, int y, int plane) const
{
//      std::cout << x << " " << y << " " << (void *)getBitMask(plane) << " " << (void *)(*_vwImagePtr)(x, y) << std::endl;
     return ((*_vwImagePtr)(x, y) & getBitMask(plane)) != 0;
}

template<typename MaskPixelT>
bool lsst::fw::MaskPixelBooleanFunc<MaskPixelT>::operator() (MaskPixelT) const {
    throw lsst::mwi::exceptions::Runtime(boost::format("You can't get here: %s:%d") % __FILE__ % __LINE__);
    return true;
}

template<typename MaskPixelT>
lsst::fw::Mask<MaskPixelT>& lsst::fw::Mask<MaskPixelT>::operator |= (const Mask<MaskPixelT>& inputMask)
{
// Need to check for identical sizes, and presence of all needed planes
    if (getCols() != inputMask.getCols() || getRows() != inputMask.getRows()) {
        throw lsst::mwi::exceptions::Runtime("Sizes do not match");
    }

    // iterate through maskplanes of inputMask, and find corresponding planes in this Mask. 
    // For the moment, require the plane assignments to be identical.   In future, allow remap

    std::map<int, std::string> inputDict = inputMask.getMaskPlaneDict();

    for (std::map<int, std::string>::iterator i = inputDict.begin(); i != inputDict.end(); i++) {
        int inputPlaneNumber = i->first;
        std::string inputPlaneName = i->second;
        if (inputPlaneName != "") {
            int thisPlaneNumber;
            try {
                getMaskPlane(inputPlaneName, thisPlaneNumber) ;
            } catch (lsst::fw::NoMaskPlane &e) {
                lsst::mwi::utils::Trace("fw.Mask", 0,
                    boost::format("%s Plane %s not present in this Mask") % e.what() % inputPlaneName);
                throw;
            }
            if (thisPlaneNumber != inputPlaneNumber) {
                lsst::mwi::utils::Trace("fw.Mask", 0,
                    boost::format("Plane %s does not have the same assignment in this Mask (%d %d) ") % inputPlaneNumber % thisPlaneNumber);
                throw lsst::mwi::exceptions::Runtime(
                    boost::format("Plane %s does not have the same assignment in this Mask (%d %d) ") % inputPlaneNumber % thisPlaneNumber);
            }
        }
    }
     
    // Now, can iterate through the MaskImages, or'ing the input pixels into this MaskImage

    for (unsigned int y = 0; y < getRows(); y++) {
        for (unsigned int x = 0; x < getCols(); x++) {
            (*_vwImagePtr)(x,y) |= inputMask(x,y);
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
void lsst::fw::Mask<MaskPixelT>::addMaskPlaneMetaData(lsst::mwi::data::DataProperty::PtrType rootPtr) {
     if( rootPtr->isNode() != true ) {
        throw lsst::mwi::exceptions::InvalidParameter( "Given DataProperty object is not a node" );
        
     }

    // First, clear existing MaskPlane metadata
    rootPtr->deleteAll( maskPlanePrefix +".*", false );

    // Add new MaskPlane metadata
    for (std::map<int, std::string>::iterator i = _maskPlaneDict.begin(); i != _maskPlaneDict.end() ; i++) {
        int planeNumber = i->first;
        std::string planeName = i->second;
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
void lsst::fw::Mask<MaskPixelT>::parseMaskPlaneMetaData(const lsst::mwi::data::DataProperty::PtrType rootPtr) {

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
void lsst::fw::Mask<MaskPixelT>::printMaskPlanes() const {
    for (std::map<int, std::string>::const_iterator i = _maskPlaneDict.begin(); i != _maskPlaneDict.end() ; i++) {
        int planeNumber = i->first;
        std::string planeName = i->second;
        std::cout << "Plane " << planeNumber << " -> " << planeName << std::endl;
    }

}

template<typename MaskPixelT>
std::map<int, std::string> lsst::fw::Mask<MaskPixelT>::getMaskPlaneDict() const {
    return _maskPlaneDict;
}

template<typename MaskPixelT>
void lsst::fw::Mask<MaskPixelT>::setOffsetRows(unsigned int offset) {
    _offsetRows = offset;
}

template<typename MaskPixelT>
void lsst::fw::Mask<MaskPixelT>::setOffsetCols(unsigned int offset)
{
    _offsetCols = offset;
}


template<typename MaskPixelT> 
const std::string lsst::fw::Mask<MaskPixelT>::maskPlanePrefix("MP_");
