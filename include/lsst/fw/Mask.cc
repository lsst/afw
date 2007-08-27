// -*- lsst-c++ -*-
// Implementations of Mask class methods
// This file can NOT be separately compiled!   It is included by Mask.h

template<typename MaskPixelT>
Mask<MaskPixelT>::Mask() :
    LsstBase(typeid(this)),
    _imagePtr(new vw::ImageView<MaskPixelT>()),
    _image(*_imagePtr),
    _numPlanesMax(8*sizeof(MaskChannelT)),
    _metaData(SupportFactory::createPropertyNode("FitsMetaData")) {

    Trace("fw.Mask", 1,
              boost::format("Number of mask planes: %d") % _numPlanesMax);

     for (int i=0; i<_numPlanesMax; i++) {
	  _planeBitMask[i] = 1 << i;
	  _planeBitMaskComplemented[i] = ~_planeBitMask[i];
	  _maskPlaneDict[i] = "";
     }

     _numPlanesUsed = 0;
     _offsetRows = 0;
     _offsetCols = 0;

}

template<class MaskPixelT>
Mask<MaskPixelT>::Mask(MaskIVwPtrT image): 
    LsstBase(typeid(this)),
    _imagePtr(image),
    _image(*_imagePtr),
    _numPlanesMax(8 * sizeof(MaskChannelT)),
    _metaData(SupportFactory::createPropertyNode("FitsMetaData")) {

    Trace("fw.Mask", 1,
              boost::format("Number of mask planes: %d") % _numPlanesMax);

     for (int i=0; i<_numPlanesMax; i++) {
	  _planeBitMask[i] = 1 << i;
	  _planeBitMaskComplemented[i] = ~_planeBitMask[i];
	  _maskPlaneDict[i] = "";
     }

     _numPlanesUsed = 0;
     _offsetRows = 0;
     _offsetCols = 0;

}

template<class MaskPixelT>
Mask<MaskPixelT>::Mask(int ncols, int nrows) :
    LsstBase(typeid(this)),
    _imagePtr(new vw::ImageView<MaskPixelT>(ncols, nrows)),
    _image(*_imagePtr),
    _numPlanesMax(8*sizeof(MaskChannelT)),
    _metaData(SupportFactory::createPropertyNode("FitsMetaData")) {

    Trace("fw.Mask", 1,
              boost::format("Number of mask planes: %d") % _numPlanesMax);

     for (int i=0; i<_numPlanesMax; i++) {
	  _planeBitMask[i] = 1 << i;
	  _planeBitMaskComplemented[i] = ~_planeBitMask[i];
	  _maskPlaneDict[i] = "";
     }

     _numPlanesUsed = 0;
     _offsetRows = 0;
     _offsetCols = 0;

    }

template<class MaskPixelT>
DataProperty::PtrType Mask<MaskPixelT>::getMetaData()
{
    return _metaData;
}

template<class MaskPixelT>
void Mask<MaskPixelT>::readFits(const string& fileName, int hdu)
{
    LSSTFitsResource<MaskPixelT> fitsRes;
    fitsRes.readFits(fileName, _image, _metaData, hdu);
    parseMaskPlaneMetaData(_metaData);
}

template<class MaskPixelT>
void Mask<MaskPixelT>::writeFits(const string& fileName)
{
    LSSTFitsResource<MaskPixelT> fitsRes;
    addMaskPlaneMetaData(_metaData);
    fitsRes.writeFits(_image, _metaData, fileName);
}

template<class MaskPixelT>
int Mask<MaskPixelT>::addMaskPlane(const string& name)
{
     int id;
     try {
         getMaskPlane(name, id);
	 return id;
     }
     catch(NoMaskPlane &e) {
        // build new entry
        if (_numPlanesUsed < _numPlanesMax) {
	    // find first entry in dict with null name
	    for(map<int, string>::iterator i=_maskPlaneDict.begin(); i != _maskPlaneDict.end(); ++i) {
	       if (i->second == "") {
		    i->second = name;
		    _numPlanesUsed++; 
		    return i->first;
	       }
	    }
            // No free space found for new plane addition
            throw OutOfPlaneSpace("No space to add new plane")
                << DataProperty("numPlanesUsed", _numPlanesUsed) 
                << DataProperty("numPlanesMax", _numPlanesMax);
        } else {
            // Max number of planes already allocated
          throw OutOfPlaneSpace("Max number of planes already used")
              << DataProperty("numPlanesUsed", _numPlanesUsed)
              << DataProperty("numPlanesMax", _numPlanesMax);
        }
    }
}

// This is a private function.  It sets the plane of the given planeId to be name
// with minimal checking.   Mainly used by setMaskPlaneMetadata

template<class MaskPixelT> 
int Mask<MaskPixelT>::addMaskPlane(string name, int planeId)
{
    if (planeId <0 || planeId >= _numPlanesMax) {
        throw;
    }

    std::string oldName = _maskPlaneDict[planeId];
    _maskPlaneDict[planeId] = name;
    if (oldName == "") _numPlanesUsed++;
    return planeId;
}

template<class MaskPixelT>
void Mask<MaskPixelT>::removeMaskPlane(const string& name)
{
     int id;
     try {
         getMaskPlane(name, id);
         clearMaskPlane(id);
         _maskPlaneDict[id] = "";
         _numPlanesUsed--;
	 return;
     }
     catch (exception &e) {
         Trace("fw.Mask", 0,
                   boost::format("%s Plane %s not present in this Mask") % e.what() % name);
         return;
     }
     
}

template<class MaskPixelT>
void Mask<MaskPixelT>::getMaskPlane(const string& name,
                                    int& plane) const {
     for(map<int, string>::const_iterator i=_maskPlaneDict.begin(); i != _maskPlaneDict.end(); ++i) {
	  if (i->second == name) {
	       plane = i->first;
	       return ;
	  }
     }
     plane = -1;
     throw NoMaskPlane("Failed to find maskPlane " + name);
}

template<class MaskPixelT>
int Mask<MaskPixelT>::getMaskPlane(const string& name) const {
    for(map<int, string>::const_iterator i=_maskPlaneDict.begin(); i != _maskPlaneDict.end(); ++i) {
        if (i->second == name) {
            return i->first;
        }
    }
    return -1;
}

template<class MaskPixelT>
bool Mask<MaskPixelT>::getPlaneBitMask(const string& name,
                                       MaskChannelT& bitMask) const {
    int plane = getMaskPlane(name);
    if (plane < 0) {
         Trace("fw.Mask", 1, boost::format("Plane %s not present in this Mask") % name);
         return false;
    }

    bitMask = _planeBitMask[plane];
    return true;
}

template<class MaskPixelT>
typename Mask<MaskPixelT>::MaskChannelT Mask<MaskPixelT>::getPlaneBitMask(const string& name) const {
    return _planeBitMask[getMaskPlane(name)];
}

template<class MaskPixelT> void Mask<MaskPixelT>::clearAllMaskPlanes() 
{
     for (int i=0; i<_numPlanesMax; i++) {
	  _maskPlaneDict[i] = "";
     }
    _numPlanesUsed = 0;
}

// clearMaskPlane(int plane) clears the bit specified by "plane" in all pixels in the mask
//
template<class MaskPixelT> void Mask<MaskPixelT>::clearMaskPlane(int plane)
{
    for (unsigned int y = 0; y < getRows(); y++) {
        for (unsigned int x = 0; x < getCols(); x++) {
	       _image(x,y) = _image(x,y) & _planeBitMaskComplemented[plane];
	  }
     }
}

// setMaskPlaneValues(int plane, list<PixelCoord> pixelList) sets the bit specified by "plane" for
// each pixel in the pixelList
//
template<class MaskPixelT> void Mask<MaskPixelT>::setMaskPlaneValues(int plane, list<PixelCoord> pixelList)
{
//      cout << "setMaskPlaneValues " << pixelList.size() << endl;
     for (list<PixelCoord>::iterator i = pixelList.begin(); i != pixelList.end(); i++) {
	  PixelCoord coord = *i;
	  _image(coord.x, coord.y) = _image(coord.x, coord.y) | _planeBitMask[plane];
// 	  cout << "Set: " << coord.x << " " << coord.y << " " << (void *)_planeBitMask[plane] << " " << _image(coord.x, coord.y) << endl;
     }
}

/**
 * setMaskPlaneValues(int plane, x0, x1, y) sets the bit specified by "plane" for pixels (x0, y) ... (x1, y)
 */
template<typename MaskPixelT>
void Mask<MaskPixelT>::setMaskPlaneValues(const int plane, const int x0, const int x1, const int y) {
    for (int x = x0; x <= x1; x++) {
        _image(x, y) |= _planeBitMask[plane];
    }
}

// setMaskPlaneValues(int plane,MaskPixelBooleanFunc selectionFunc ) sets the bit specified by "plane"
// for each pixel for which selectionFunc(pixel) returns true
//
template<class MaskPixelT> void Mask<MaskPixelT>::setMaskPlaneValues(int plane, MaskPixelBooleanFunc<MaskPixelT> selectionFunc)
{
    //  Should check plane for legality here...

    for (unsigned int y = 0; y < getRows(); y++) {
        for (unsigned int x = 0; x < getCols(); x++) {
            if (selectionFunc(_image(x,y)) == true) {
                _image(x,y) = _image(x,y) | _planeBitMask[plane];
            }
          }
    }
}

// countMask(MaskPixelBooleanFunc testFunc, BBox2i maskRegion) returns the number of pixels
// within maskRegion for which testFunc(pixel) returns true
//
// PROBABLY WANT maskRegion to default to whole Mask

template<class MaskPixelT>
int Mask<MaskPixelT>::countMask(MaskPixelBooleanFunc<MaskPixelT>& testFunc,
                                const BBox2i maskRegion) const {
    int count = 0;
    Vector<int32, 2> ulCorner = maskRegion.min();
    Vector<int32, 2> lrCorner = maskRegion.max();

    for (int y=ulCorner[1]; y<lrCorner[1]; y++) {
        for (int x=ulCorner[0]; x<lrCorner[0]; x++) {
            if (testFunc(_image(x,y)) == true) {
                count += 1;
            }
          }
    }
     return count;
}

template<class MaskPixelT>
typename Mask<MaskPixelT>::MaskPtrT Mask<MaskPixelT>::getSubMask(const vw::BBox2i maskRegion) const {

    // Check that maskRegion is completely inside the mask
    
    BBox2i maskBoundary(0, 0, getCols(), getRows());
    if (!maskBoundary.contains(maskRegion)) {
        throw lsst::mwi::exceptions::InvalidParameter(boost::format("getSubMask region not contained within Mask"));
    }

    MaskIVwPtrT croppedMask(new MaskIVwT());
    *croppedMask = copy(crop(_image, maskRegion));
    MaskPtrT newMask(new Mask<MaskPixelT>(croppedMask));
    Vector<int, 2> bboxOffset = maskRegion.min();
    newMask->setOffsetRows(bboxOffset[1] + _offsetRows);
    newMask->setOffsetCols(bboxOffset[0] + _offsetCols);
    return newMask;
}

// Given a Mask, insertMask, place it into this Mask as directed by maskRegion.
// An exception is generated if maskRegion is not of the same size as insertMask.
// Maybe generate an exception if offsets are not consistent?
//
template<class MaskPixelT>
void Mask<MaskPixelT>::replaceSubMask(const BBox2i maskRegion, MaskPtrT insertMask)
{
    try {
        crop(_image, maskRegion) = insertMask->_image;
    } catch (exception eex) {
        throw Exception(std::string("in ") + __func__);
    } 
}

template<class MaskPixelT> typename Mask<MaskPixelT>::MaskChannelT Mask<MaskPixelT>::operator ()(int x, int y) const
{
//      cout << x << " " << y << " " << (void *)_image(x, y) << endl;
     return _image(x, y);
}

template<class MaskPixelT> bool Mask<MaskPixelT>::operator ()(int x, int y, int plane) const
{
//      cout << x << " " << y << " " << (void *)_planeBitMask[plane] << " " << (void *)_image(x, y) << endl;
     return (_image(x, y) & _planeBitMask[plane]) != 0;
}

template<typename MaskPixelT>
bool MaskPixelBooleanFunc<MaskPixelT>::operator() (MaskPixelT) const {
    throw Exception(boost::format("You can't get here: %s:%d") % __FILE__ % __LINE__);
    return true;
}

template<class MaskPixelT> Mask<MaskPixelT>&  Mask<MaskPixelT>::operator |= (const Mask<MaskPixelT>& inputMask)
{
// Need to check for identical sizes, and presence of all needed planes
    if (getCols() != inputMask.getCols() || getRows() != inputMask.getRows()) {
        throw;
    }

    // iterate through maskplanes of inputMask, and find corresponding planes in this Mask. 
    // For the moment, require the plane assignments to be identical.   In future, allow remap

    map<int, std::string> inputDict = inputMask.getMaskPlaneDict();

    for (map<int, std::string>::iterator i = inputDict.begin(); i != inputDict.end(); i++) {
        int inputPlaneNumber = i->first;
        string inputPlaneName = i->second;
        if (inputPlaneName != "") {
            int thisPlaneNumber;
            try {
                getMaskPlane(inputPlaneName, thisPlaneNumber) ;
            }
            catch (NoMaskPlane &e) {
                Trace("fw.Mask", 0,
                          boost::format("%s Plane %s not present in this Mask") % e.what() % inputPlaneName);
                throw;
            }
            if (thisPlaneNumber != inputPlaneNumber) {
                Trace("fw.Mask", 0,
                          boost::format("Plane %s does not have the same assignment in this Mask (%d %d) ") % inputPlaneNumber % thisPlaneNumber);
                throw;
            }
        }
    }
     
    // Now, can iterate through the MaskImages, or'ing the input pixels into this MaskImage

    for (unsigned int y = 0; y < getRows(); y++) {
        for (unsigned int x = 0; x < getCols(); x++) {
            _image(x,y) |= inputMask(x,y);
        }
    }

    return *this;
}

/** Given a DataProperty, replace any existing MaskPlane assignments with the current ones.
  *
  * \throw Throws lsst::fw::InvalidParameter if given DataProperty is not a node
  */
template<class MaskPixelT>
void Mask<MaskPixelT>::addMaskPlaneMetaData(DataProperty::PtrType rootPtr) {
     if( rootPtr->isNode() != true ) {
        throw lsst::fw::InvalidParameter( "Given DataProperty object is not a node" );
        
     }

    // First, clear existing MaskPlane metadata
    rootPtr->deleteAll( maskPlanePrefix +".*", false );

    // Add new MaskPlane metadata
    for (map<int, std::string>::iterator i = _maskPlaneDict.begin(); i != _maskPlaneDict.end() ; i++) {
        int planeNumber = i->first;
        std::string planeName = i->second;
        if (planeName != "") {
            rootPtr->addProperty(
                DataProperty::PtrType(new DataProperty(Mask::maskPlanePrefix + planeName, planeNumber)));
        }
    }
}


/** Given a DataProperty that contains the MaskPlane assignments setup the MaskPlanes.
  */

template<class MaskPixelT>
void Mask<MaskPixelT>::parseMaskPlaneMetaData(const DataProperty::PtrType rootPtr) {

    DataProperty::iteratorRangeType range = rootPtr->searchAll( maskPlanePrefix +".*" );
    if (std::distance(range.first, range.second) == 0) {
        return;
    }

    // Clear all existing MaskPlanes

    clearAllMaskPlanes();

    // Iterate through matching keyWords
    DataProperty::ContainerIteratorType iter;
    for( iter = range.first; iter != range.second; iter++ ) {
        DataProperty::PtrType dpPtr = *iter;
        // split off the "MP_" to get the planeName
        std::string keyWord = dpPtr->getName();
        std::string planeName = keyWord.substr(maskPlanePrefix.size());
        // will throw an exception if the found item does not contain const int
        int planeId = boost::any_cast<const int>(dpPtr->getValue());
        addMaskPlane(planeName, planeId);
    }

}

template<class MaskPixelT> void Mask<MaskPixelT>::printMaskPlanes() const {

        for (map<int, std::string>::const_iterator i = _maskPlaneDict.begin(); i != _maskPlaneDict.end() ; i++) {
        int planeNumber = i->first;
        std::string planeName = i->second;
        cout << "Plane " << planeNumber << " -> " << planeName << endl;
        }

}

template<class MaskPixelT> map<int, std::string> Mask<MaskPixelT>::getMaskPlaneDict() const {
    return _maskPlaneDict;
}

template<class MaskPixelT> void Mask<MaskPixelT>::setOffsetRows(unsigned int offset)
{
    _offsetRows = offset;
}

template<class MaskPixelT> void Mask<MaskPixelT>::setOffsetCols(unsigned int offset)
{
    _offsetCols = offset;
}


template<class MaskPixelT> 
const std::string Mask<MaskPixelT>::maskPlanePrefix("MP_");
