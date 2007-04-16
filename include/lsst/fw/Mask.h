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
#include "lsst/fw/LsstBase.h"
#include "lsst/fw/Exception.h"
#include "lsst/fw/DataProperty.h"

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
        virtual bool operator () (MaskPixelT) const;
        virtual ~MaskPixelBooleanFunc() {}
    protected:
        Mask<MaskPixelT>& _mask;
    };
    
    template<typename MaskPixelT>
    class Mask : private fw::LsstBase {
    public:
        typedef typename PixelChannelType<MaskPixelT>::type MaskChannelT;
        typedef ImageView<MaskPixelT> MaskIVwT;
        typedef boost::shared_ptr<Mask<MaskPixelT> > MaskPtrT;
        typedef boost::shared_ptr<MaskIVwT> MaskIVwPtrT;
        
        Mask();

        Mask(MaskIVwPtrT image);

        Mask(int nCols, int nRows);
        
        int addMaskPlane(const string& name);
        
        void removeMaskPlane(const string& name);

        void getMaskPlane(const string& name, int& plane) const;

        bool getPlaneBitMask(const string& name,
                             MaskChannelT& bitMask) const;
        
        void clearMaskPlane(int plane);

        void clearAllMaskPlanes();
        
        void setMaskPlaneValues(int plane, list<PixelCoord> pixelList);
        
        void setMaskPlaneValues(int plane,
                                MaskPixelBooleanFunc<MaskPixelT> selectionFunc);

        void parseMaskPlaneMetaData(const DataProperty::DataPropertyPtrT);

        void addMaskPlaneMetaData(DataProperty::DataPropertyPtrT);

        int countMask(MaskPixelBooleanFunc<MaskPixelT>& testFunc,
                      const BBox2i maskRegion) const;

        MaskPtrT getSubMask(const BBox2i maskRegion) const;

        void replaceSubMask(const BBox2i maskRegion, MaskPtrT insertMask);

        MaskChannelT operator ()(int x, int y) const;

        bool operator ()(int x, int y, int plane) const;

        Mask<MaskPixelT>& operator |= (const Mask<MaskPixelT>& inputMask);

        int getImageCols() const;

        int getImageRows() const;

        MaskIVwPtrT getIVwPtr() const;

        map<int, std::string> getMaskPlaneDict() const;

        void printMaskPlanes() const;

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
        static const std::string maskPlanePrefix;

        int addMaskPlane(string name, int plane);
    };
  
#include "Mask.cc"  


} // namespace lsst

#endif // LSST_MASK_H


