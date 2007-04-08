// -*- lsst-c++ -*-
// Implementations of Mask class methods
// This file can NOT be separately compiled!   It is included by Mask.h

#include <lsst/fw/Trace.h>

template<typename MaskPixelT>
Mask<MaskPixelT>::Mask() :
    fw::LsstBase(typeid(this)),
    _imagePtr(new vw::ImageView<MaskPixelT>()),
    _image(*_imagePtr),
    _numPlanesMax(8*sizeof(MaskChannelT)) {
}

template<class MaskPixelT>
Mask<MaskPixelT>::Mask(MaskIVwPtrT image): 
    fw::LsstBase(typeid(this)),
    _imagePtr(image),
    _image(*_imagePtr),
    _numPlanesMax(8 * sizeof(MaskChannelT)) {
    _imageRows = _image.rows();
    _imageCols = _image.cols();

    fw::Trace::trace("fw.Mask", 1,
                 boost::format("Number of mask planes: %d") % _numPlanesMax);

     for (int i=0; i<_numPlanesMax; i++) {
	  _planeBitMask[i] = 1 << i;
	  _planeBitMaskComplemented[i] = ~_planeBitMask[i];
	  _maskPlaneDict[i] = "";
     }

     _numPlanesUsed = 0;

}

template<class MaskPixelT>
Mask<MaskPixelT>::Mask(int ncols, int nrows) :
    fw::LsstBase(typeid(this)),
    _imagePtr(new vw::ImageView<MaskPixelT>(ncols, nrows)),
    _image(*_imagePtr),
    _numPlanesMax(8*sizeof(MaskChannelT)) {
    _imageRows = _image.rows();
    _imageCols = _image.cols();

    fw::Trace::trace("fw.Mask", 1,
                     boost::format("Number of mask planes: %d") % _numPlanesMax);

     for (int i=0; i<_numPlanesMax; i++) {
	  _planeBitMask[i] = 1 << i;
	  _planeBitMaskComplemented[i] = ~_planeBitMask[i];
	  _maskPlaneDict[i] = "";
     }

     _numPlanesUsed = 0;

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
#if 0
            std::string s("No space to add new plane");
            DataPropertyPtr propertyList(new DataProperty("root",(int)0));
            DataPropertyPtr aProperty(new DataProperty("numPlanesUsed",(int)_numPlanesUsed));
            propertyList->addProperty(aProperty);
            DataPropertyPtr bProperty(new DataProperty("numPlanesMax",(int)_numPlanesMax));
            propertyList->addProperty(bProperty);
            propertyList->print(">>\t");
            throw OutOfPlaneSpace(s, propertyList);
#else
            throw OutOfPlaneSpace("No space to add new plane")
                << DataProperty("numPlanesUsed", _numPlanesUsed) 
                << DataProperty("numPlanesMax", _numPlanesMax);
#endif
        } else {
            // Max number of planes already allocated
#if 0
          std::string s = "Max number of planes already used";
          DataPropertyPtr propertyList(new DataProperty("root",(int)0));
          DataPropertyPtr aProperty(new DataProperty("numPlanesUsed",(int)_numPlanesUsed));
          propertyList->addProperty(aProperty);
          DataPropertyPtr bProperty(new DataProperty("numPlanesMax",(int)_numPlanesMax));
          propertyList->addProperty(bProperty);
          propertyList->print(">>\t");
          OutOfPlaneSpace oops(OutOfPlaneSpace(s, propertyList));
          cout << "Throwing oops " << oops.propertyList().use_count() << " | "
               << static_cast<Citizen *>(oops.propertyList().get())->repr()
               << endl;
          throw oops;
#else
          throw OutOfPlaneSpace("Max number of planes already used")
              << DataProperty("numPlanesUsed", _numPlanesUsed)
              << DataProperty("numPlanesMax", _numPlanesMax);
#endif
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
         fw::Trace::trace("fw.Mask", 0,
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
bool Mask<MaskPixelT>::getPlaneBitMask(const string& name,
                                       MaskChannelT& bitMask) const {
    int plane;
    try {
        getMaskPlane(name, plane);
    }
    catch (exception &e) {
         fw::Trace::trace("fw.Mask", 0,
                          boost::format("%s Plane %s not present in this Mask") % e.what() % name);
         return false;
    }

    bitMask = _planeBitMask[plane];
    return true;
}

template<class MaskPixelT> void Mask<MaskPixelT>::clearAllMaskPlanes() {
    _maskPlaneDict.clear();
    _numPlanesUsed = 0;
}

// clearMaskPlane(int plane) clears the bit specified by "plane" in all pixels in the mask
//
template<class MaskPixelT> void Mask<MaskPixelT>::clearMaskPlane(int plane)
{
     for (int y=0; y<_imageRows; y++) {
	  for (int x=0; x<_imageCols; x++) {
	       _image(x,y).v() = _image(x,y).v() & _planeBitMaskComplemented[plane];
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
	  _image(coord.x, coord.y).v() = _image(coord.x, coord.y).v() | _planeBitMask[plane];
// 	  cout << "Set: " << coord.x << " " << coord.y << " " << (void *)_planeBitMask[plane] << " " << _image(coord.x, coord.y) << endl;
     }
}

// setMaskPlaneValues(int plane,MaskPixelBooleanFunc selectionFunc ) sets the bit specified by "plane"
// for each pixel for which selectionFunc(pixel) returns true
//
template<class MaskPixelT> void Mask<MaskPixelT>::setMaskPlaneValues(int plane, MaskPixelBooleanFunc<MaskPixelT> selectionFunc)
{
    //  Should check plane for legality here...

    for (int y=0; y<_imageRows; y++) {
        for (int x=0; x<_imageCols; x++) {
            if (selectionFunc(_image(x,y)) == true) {
                _image(x,y).v() = _image(x,y).v() | _planeBitMask[plane];
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

    MaskIVwPtrT croppedMask(new MaskIVwT());
    *croppedMask = copy(crop(_image, maskRegion));
    MaskPtrT newMask(new Mask<MaskPixelT>(croppedMask));
    return newMask;
}

// Given a Mask, insertMask, place it into this Mask as directed by maskRegion.
// An exception is generated if maskRegion is not of the same size as insertMask.
//
template<class MaskPixelT>
void Mask<MaskPixelT>::replaceSubMask(const BBox2i maskRegion, MaskPtrT insertMask)
{
    try {
        crop(_image, maskRegion) = insertMask->_image;
    } catch (exception eex) {
        throw lsst::Exception(std::string("in ") + __func__);
    } 
}

template<class MaskPixelT> typename Mask<MaskPixelT>::MaskChannelT Mask<MaskPixelT>::operator ()(int x, int y) const
{
//      cout << x << " " << y << " " << (void *)_image(x, y).v() << endl;
     return _image(x, y).v();
}

template<class MaskPixelT> bool Mask<MaskPixelT>::operator ()(int x, int y, int plane) const
{
//      cout << x << " " << y << " " << (void *)_planeBitMask[plane] << " " << (void *)_image(x, y).v() << endl;
     return (_image(x, y).v() & _planeBitMask[plane]) != 0;
}

template<typename MaskPixelT>
bool MaskPixelBooleanFunc<MaskPixelT>::operator() (MaskPixelT) const {
    throw lsst::Exception(boost::format("You can't get here: %s:%d") % __FILE__ % __LINE__);
    return true;
}

template<class MaskPixelT> Mask<MaskPixelT>&  Mask<MaskPixelT>::operator |= (const Mask<MaskPixelT>& inputMask)
{
// Need to check for identical sizes, and presence of all needed planes
    if (_imageCols != inputMask.getImageCols() || _imageRows != inputMask.getImageRows()) {
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
                fw::Trace::trace("fw.Mask", 0,
                                 boost::format("%s Plane %s not present in this Mask") % e.what() % inputPlaneName);
                throw;
            }
            if (thisPlaneNumber != inputPlaneNumber) {
                fw::Trace::trace("fw.Mask", 0,
                                 boost::format("Plane %s does not have the same assignment in this Mask (%d %d) ") % inputPlaneNumber % thisPlaneNumber);
                throw;
            }
        }
    }
     
    // Now, can iterate through the MaskImages, or'ing the input pixels into this MaskImage

    for (int y=0; y<_imageRows; y++) {
        for (int x=0; x<_imageCols; x++) {
            _image(x,y).v() |= inputMask(x,y);
        }
    }

    return *this;
}

// Build a DataProperty that describes the MaskPlane assignments, and return the pointer to it.

template<class MaskPixelT>
DataProperty::DataPropertyPtrT Mask<MaskPixelT>::getMaskPlaneMetaData() {

    DataProperty::DataPropertyPtrT rootPtr(new DataProperty("MaskPlaneMetaData", 0));

    // iterate over MaskPlanes
    for (map<int, std::string>::iterator i = _maskPlaneDict.begin(); i != _maskPlaneDict.end() ; i++) {
        int planeNumber = i->first;
        std::string planeName = i->second;
        if (planeName != "") {
            rootPtr->addProperty(DataProperty::DataPropertyPtrT(new DataProperty(planeName, planeNumber)));
        }
    }

    return rootPtr;
}

// Given a DataProperty that describes the MaskPlane assignments setup the MaskPlanes.

template<class MaskPixelT> void Mask<MaskPixelT>::setMaskPlaneMetaData(const DataProperty::DataPropertyPtrT rootPtr) {

    if (rootPtr->getName() != "MaskPlaneMetaData") {
        throw;
    }

    // Clear all existing MaskPlanes

    clearAllMaskPlanes();

    // Iterate through contents of DataProperty

    const DataProperty::DataPropertyContainerT contents = rootPtr->getContents();

    for (DataProperty::DataPropertyContainerT::const_iterator i = contents.begin(); i != contents.end(); i++) {
        DataProperty::DataPropertyPtrT dpPtr = *i;
        addMaskPlane(dpPtr->getName(), boost::any_cast<const int>(dpPtr->getValue()));
    }

}

template<class MaskPixelT> void Mask<MaskPixelT>::printMaskPlanes() const {

        for (map<int, std::string>::const_iterator i = _maskPlaneDict.begin(); i != _maskPlaneDict.end() ; i++) {
        int planeNumber = i->first;
        std::string planeName = i->second;
        cout << "Plane " << planeNumber << " -> " << planeName << endl;
        }

}

template<class MaskPixelT> int Mask<MaskPixelT>::getImageCols() const {
    return _imageCols;
}

template<class MaskPixelT> int Mask<MaskPixelT>::getImageRows() const {
    return _imageRows;
}

template<class MaskPixelT> typename Mask<MaskPixelT>::MaskIVwPtrT Mask<MaskPixelT>::getIVwPtr() const {
    return _imagePtr;
    // did this increment reference count or not....and does this violate const??
}

template<class MaskPixelT> map<int, std::string> Mask<MaskPixelT>::getMaskPlaneDict() const {
    return _maskPlaneDict;
}
