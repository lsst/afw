// -*- lsst-c++ -*-
///////////////////////////////////////////////////////////
//  Mask.h
//  Implementation of the Class Mask
//  Created on:      09-Feb-2007 15:57:46
//  Original author: Tim Axelrod
///////////////////////////////////////////////////////////

#ifndef LSST_MASK_H
#define LSST_MASK_H

#include <vw/Image.h>
#include <list>
#include <map>
#include <string>

#include "PixelRegion.h"

using namespace vw;
using namespace std;

namespace lsst {


    template<class MaskPixelT>
    class Mask
    {
    public:
        typedef bool (*MaskPixelBooleanFunc) (MaskPixelT);
        typedef typename PixelChannelType<MaskPixelT>::type MaskChannelT;
        
        Mask();

        Mask(ImageView<MaskPixelT>& image);
        
        int addMaskPlane(string name);
        
        void removeMaskPlane(string name);

        bool findMaskPlane(string name, int& plane);
        
        void clearMaskPlane(int plane);
        
        void setMaskPlaneValues(int plane, list<PixelCoord> pixelList);
        
        void setMaskPlaneValues(int plane, MaskPixelBooleanFunc selectionFunc);
        
        int countMask(MaskPixelBooleanFunc testFunc, PixelRegion maskRegion);

        // do we want an operator (x,y, iplane) to give the mask values?

        MaskChannelT operator ()(int x, int y);

        bool operator ()(int x, int y, int plane);

//         virtual ~Mask();

    private:
        ImageView<MaskPixelT>& _image;
        int _imageRows;
        int _imageCols;
        map<int, std::string> _maskPlaneDict;
        const int _numPlanesMax;
        int _numPlanesUsed;
        MaskChannelT _planeBitMask[8 * sizeof(MaskChannelT)];
        MaskChannelT _planeBitMaskComplemented[8 * sizeof(MaskChannelT)];

    };
  
#include "Mask.cc"  


} // namespace lsst

#endif // LSST_MASK_H


