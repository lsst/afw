// -*- lsst-c++ -*-
// Implementations of Mask class methods
// This file can NOT be separately compiled!   It is included by Mask.h


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

     cout << "Number of mask planes: " << _numPlanesMax << endl;

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

     cout << "Number of mask planes: " << _numPlanesMax << endl;

     for (int i=0; i<_numPlanesMax; i++) {
	  _planeBitMask[i] = 1 << i;
	  _planeBitMaskComplemented[i] = ~_planeBitMask[i];
	  _maskPlaneDict[i] = "";
     }

     _numPlanesUsed = 0;

    }


template<class MaskPixelT> int Mask<MaskPixelT>::addMaskPlane(string name)
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
            std::string s("No space to add new plane");
            DataPropertyPtr propertyList(new DataProperty("root",(int)0));
            DataPropertyPtr aProperty(new DataProperty("numPlanesUsed",(int)_numPlanesUsed));
            propertyList->addProperty(aProperty);
            DataPropertyPtr bProperty(new DataProperty("numPlanesMax",(int)_numPlanesMax));
            propertyList->addProperty(bProperty);
            propertyList->print();
            OutOfPlaneSpace oops(OutOfPlaneSpace(s, propertyList));
            throw oops;

        } else {
            // Max number of planes already allocated
          std::string s = "Max number of planes already used";
          DataPropertyPtr propertyList(new DataProperty("root",(int)0));
          DataPropertyPtr aProperty(new DataProperty("numPlanesUsed",(int)_numPlanesUsed));
          propertyList->addProperty(aProperty);
          DataPropertyPtr bProperty(new DataProperty("numPlanesMax",(int)_numPlanesMax));
          propertyList->addProperty(bProperty);
          propertyList->print();
          OutOfPlaneSpace oops(OutOfPlaneSpace(s, propertyList));
          throw oops;
        }
    }
}

template<class MaskPixelT> void Mask<MaskPixelT>::removeMaskPlane(string name)
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
         cout << e.what() << "Plane " << name << " not present in this Mask" << endl;
         return;
     }
     
}

template<class MaskPixelT> void Mask<MaskPixelT>::getMaskPlane(string name, int& plane)
{
     for(map<int, string>::iterator i=_maskPlaneDict.begin(); i != _maskPlaneDict.end(); ++i) {
	  if (i->second == name) {
	       plane = i->first;
	       return ;
	  }
     }
     plane = -1;
     throw NoMaskPlane("Failed miserably");
}

template<class MaskPixelT> bool Mask<MaskPixelT>::getPlaneBitMask(string name, MaskChannelT& bitMask)
{
    int plane;
    try {
        getMaskPlane(name, plane);
    }
    catch (exception &e) {
         cout << e.what() << "Plane " << name << " not present in this Mask" << endl;
         return false;
    }

    bitMask = _planeBitMask[plane];
    return true;
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

template<class MaskPixelT> int Mask<MaskPixelT>::countMask(MaskPixelBooleanFunc<MaskPixelT>& testFunc, BBox2i maskRegion)
{
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

template<class MaskPixelT>  typename Mask<MaskPixelT>::MaskPtrT Mask<MaskPixelT>::getSubMask(vw::BBox2i maskRegion) {

    MaskIVwPtrT croppedMask(new MaskIVwT());
    *croppedMask = copy(crop(_image, maskRegion));
    MaskPtrT newMask(new Mask<MaskPixelT>(croppedMask));
    return newMask;
}

// Given a Mask, insertMask, place it into this Mask as directed by maskRegion.
// An exception is generated if maskRegion is not of the same size as insertMask.
//
template<class MaskPixelT> void Mask<MaskPixelT>::replaceSubMask(BBox2i maskRegion, Mask<MaskPixelT>& insertMask)
{
    crop(_image, maskRegion) = insertMask._image;
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
bool MaskPixelBooleanFunc<MaskPixelT>::operator() (MaskPixelT) {
    abort();
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
                cout << e.what() << "Plane " << inputPlaneName << " not present in this Mask" << endl;
                throw;
            }
            if (thisPlaneNumber != inputPlaneNumber) {
                cout << "Plane " << inputPlaneName << " does not have the same assignment in this Mask (" 
                     << inputPlaneNumber << " " << thisPlaneNumber << ")" << endl;
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
