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
#include <vw/Math/BBox.h>
#include <boost/shared_ptr.hpp>
#include <list>
#include <map>
#include <string>
#include "lsst/Exception.h"


using namespace vw;
using namespace std;

namespace lsst {

    class PixelCoord {
    public:
        PixelCoord(int x = 0, int y = 0) : x(x), y(y) {}
        ~PixelCoord() {}
        int x;
        int y;
    };

    template<typename MaskPixelT> class Mask;

    template <typename MaskPixelT> class MaskPixelBooleanFunc {
    public:
        MaskPixelBooleanFunc(Mask<MaskPixelT>& m) : _mask(m) {}
        virtual bool operator () (MaskPixelT);
        virtual ~MaskPixelBooleanFunc() {}
    protected:
        Mask<MaskPixelT>& _mask;
    };
    
    template<typename MaskPixelT>
    class Mask
    {
    public:
        typedef typename PixelChannelType<MaskPixelT>::type MaskChannelT;
        typedef ImageView<MaskPixelT> MaskIVwT;
        typedef boost::shared_ptr<Mask<MaskPixelT> > MaskPtrT;
        typedef boost::shared_ptr<MaskIVwT> MaskIVwPtrT;

        
        Mask();

        Mask(MaskIVwPtrT image);

        Mask(int nCols, int nRows);
        
        int addMaskPlane(string name);
        
        void removeMaskPlane(string name);

        void getMaskPlane(string name, int& plane);

        bool getPlaneBitMask(string name, MaskChannelT& bitMask);
        
        void clearMaskPlane(int plane);
        
        void setMaskPlaneValues(int plane, list<PixelCoord> pixelList);
        
        void setMaskPlaneValues(int plane, MaskPixelBooleanFunc<MaskPixelT> selectionFunc);
        
        int countMask(MaskPixelBooleanFunc<MaskPixelT>& testFunc, BBox2i maskRegion);

        MaskPtrT getSubMask(BBox2i maskRegion);

        void replaceSubMask(BBox2i maskRegion, Mask<MaskPixelT>& insertMask);

        MaskChannelT operator ()(int x, int y) const;

        bool operator ()(int x, int y, int plane) const;

        Mask<MaskPixelT>& operator |= (const Mask<MaskPixelT>& inputMask);

        int getImageCols() const;

        int getImageRows() const;

        MaskIVwPtrT getIVwPtr() const;

        map<int, std::string> getMaskPlaneDict() const;

//         virtual ~Mask();

    private:
        MaskIVwPtrT _imagePtr;
        MaskIVwT& _image;
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


