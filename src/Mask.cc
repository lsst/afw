// -*- lsst-c++ -*-
// Implementations of Mask class methods
// This file can NOT be separately compiled!   It is included by Mask.h

template<class MaskPixelT> Mask<MaskPixelT>::Mask(ImageView<MaskPixelT>& image): 
     _image(image),
     _numPlanesMax(8 * sizeof(MaskChannelT))
{
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
     if (findMaskPlane(name, id) == true) {
	  // raise exception?
	  return id;
     }

     if (_numPlanesUsed < _numPlanesMax) {
	  // find first entry in dict with null name
	  for(map<int, string>::iterator i=_maskPlaneDict.begin(); i != _maskPlaneDict.end(); ++i) {
	       if (i->second == "") {
		    i->second = name;
		    _numPlanesUsed++; 
		    return i->first;
	       }
	  }
	  // raise exception?
	  return -1;
     } else {
	  // raise exception?
	  return -1;
     }
}

// Should removeMaskPlane clear the plane before removal?

template<class MaskPixelT> void Mask<MaskPixelT>::removeMaskPlane(string name)
{
     int id;
     if (findMaskPlane(name, id) == false) {
	  // raise exception?
	  return;
     }

     _maskPlaneDict[id] = "";
     _numPlanesUsed--;
     
}

template<class MaskPixelT> bool Mask<MaskPixelT>::findMaskPlane(string name, int& plane)
{
     for(map<int, string>::iterator i=_maskPlaneDict.begin(); i != _maskPlaneDict.end(); ++i) {
	  if (i->second == name) {
	       plane = i->first;
	       return true;
	  }
     }
     plane = -1;
     return false;
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
template<class MaskPixelT> void Mask<MaskPixelT>::setMaskPlaneValues(int plane, MaskPixelBooleanFunc selectionFunc)
{
    //  Should check plane for legality here...

    for (int y=0; y<_imageRows; y++) {
        for (int x=0; x<_imageCols; x++) {
            if ((*selectionFunc)(_image(x,y).v()) == true) {
                _image(x,y).v() = _image(x,y).v() | _planeBitMask[plane];
            }
          }
    }
}

// countMask(MaskPixelBooleanFunc testFunc, BBox2i maskRegion) returns the number of pixels
// within maskRegion for which testFunc(pixel) returns true
//
// PROBABLY WANT maskRegion to default to whole Mask

template<class MaskPixelT> int Mask<MaskPixelT>::countMask(MaskPixelBooleanFunc testFunc, BBox2i maskRegion)
{
    int count = 0;
    Vector<int32, 2> ulCorner = maskRegion.min();
    Vector<int32, 2> lrCorner = maskRegion.max();

    for (int y=ulCorner[1]; y<lrCorner[1]; y++) {
        for (int x=ulCorner[0]; x<lrCorner[0]; x++) {
            if ((*testFunc)(_image(x,y).v()) == true) {
                count += 1;
            }
          }
    }
     return count;
}

template<class MaskPixelT>  Mask<MaskPixelT>* Mask<MaskPixelT>::getSubMask(BBox2i maskRegion)
{
    // NOTE - later, this will need to use smart_ptr through Citizen 

    MaskImageT *croppedMask = new MaskImageT;
    *croppedMask = copy(crop(_image, maskRegion));
    Mask<MaskPixelT> *newMask = new Mask<MaskPixelT>(*croppedMask);
    return newMask;
}

// Given a Mask, insertMask, place it into this Mask as directed by maskRegion.
// An exception is generated if maskRegion is not of the same size as insertMask.
//
template<class MaskPixelT> void Mask<MaskPixelT>::replaceSubMask(BBox2i maskRegion, Mask<MaskPixelT>& insertMask)
{
    crop(_image, maskRegion) = insertMask._image;
}

template<class MaskPixelT> typename Mask<MaskPixelT>::MaskChannelT Mask<MaskPixelT>::operator ()(int x, int y)
{
//      cout << x << " " << y << " " << (void *)_image(x, y).v() << endl;
     return _image(x, y).v();
}

template<class MaskPixelT> bool Mask<MaskPixelT>::operator ()(int x, int y, int plane)
{
//      cout << x << " " << y << " " << (void *)_planeBitMask[plane] << " " << (void *)_image(x, y).v() << endl;
     return (_image(x, y).v() & _planeBitMask[plane]) != 0;
}
