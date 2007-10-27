// -*- lsst-c++ -*-
///////////////////////////////////////////////////////////
//  Mask.h
//  Implementation of the Class Mask
//  Created on:      09-Feb-2007 15:57:46
//  Original author: Tim Axelrod
///////////////////////////////////////////////////////////

#ifndef LSST_MASK_H
#define LSST_MASK_H

#include <list>
#include <map>
#include <string>

#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>
#include <vw/Image.h>
#include <vw/Math/BBox.h>

#include "lsst/mwi/data/LsstBase.h"
#include "lsst/mwi/exceptions.h"
#include "lsst/mwi/data/DataProperty.h"
#include "lsst/mwi/utils/Trace.h"
#include "lsst/mwi/data/SupportFactory.h"
#include "lsst/fw/LSSTFitsResource.h"
#include "lsst/fw/fwExceptions.h"

namespace lsst {
namespace fw {
    // all masks will be instantiated with the same pixel type
    typedef boost::uint16_t maskPixelType;

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
    class Mask : private lsst::mwi::data::LsstBase {
    public:
        typedef typename PixelChannelType<MaskPixelT>::type MaskChannelT;
        typedef vw::ImageView<MaskPixelT> MaskIVwT;
        typedef boost::shared_ptr<Mask<MaskPixelT> > MaskPtrT;
        typedef boost::shared_ptr<MaskIVwT> MaskIVwPtrT;
        typedef typename vw::ImageView<MaskPixelT>::pixel_accessor pixel_accessor;
        
        explicit Mask();
        
        explicit Mask(MaskIVwPtrT vwImagePtr);
        
        explicit Mask(int nCols, int nRows);

        Mask<MaskPixelT>& operator=(const Mask<MaskPixelT>& rhs);
        
        void readFits(const std::string& fileName, int hdu=0);
        
        void writeFits(const std::string& fileName);
        
        lsst::mwi::data::DataProperty::PtrType getMetaData();
        
        int addMaskPlane(const std::string& name);
        
        void removeMaskPlane(const std::string& name);
        
#if !defined(SWIG)
        void getMaskPlane(const std::string& name, int& plane) const;
#endif
        int getMaskPlane(const std::string& name) const;
        
        bool getPlaneBitMask(const std::string& name,
                             MaskChannelT& bitMask) const;
        MaskChannelT getPlaneBitMask(const std::string& name) const;
        
        void clearMaskPlane(int plane);
        
        void clearAllMaskPlanes();
        
        void setMaskPlaneValues(int plane, std::list<PixelCoord> pixelList);
        void setMaskPlaneValues(const int plane, const int x0, const int x1, const int y);
        void setMaskPlaneValues(int plane, MaskPixelBooleanFunc<MaskPixelT> selectionFunc);
        
        void parseMaskPlaneMetaData(const lsst::mwi::data::DataProperty::PtrType);
        
        void addMaskPlaneMetaData(lsst::mwi::data::DataProperty::PtrType);
        
        int countMask(MaskPixelBooleanFunc<MaskPixelT>& testFunc,
                      const vw::BBox2i maskRegion) const;
        
        MaskPtrT getSubMask(const vw::BBox2i maskRegion) const;
        
        void replaceSubMask(const vw::BBox2i maskRegion, MaskPtrT insertMask);
        
        pixel_accessor origin() const { return getIVwPtr()->origin(); }
        
        MaskChannelT operator ()(int x, int y) const;
        
        bool operator ()(int x, int y, int plane) const;
        
        Mask<MaskPixelT>& operator |= (const Mask<MaskPixelT>& inputMask);
        
        unsigned int getCols() const { return _vwImagePtr->cols(); }
        unsigned int getRows() const { return _vwImagePtr->rows(); }
        unsigned int getOffsetCols() const { return _offsetCols; }
        unsigned int getOffsetRows() const { return _offsetRows; }

        MaskIVwPtrT getIVwPtr() const {
            return _vwImagePtr; // did this increment reference count or not....and does this violate const??
        }                

        MaskIVwT& getIVw() const { return *_vwImagePtr; }

        int getNumPlanesMax() const { return 8 * sizeof(MaskChannelT); }
        int getNumPlanesUsed() const { return _numPlanesUsed; }

        std::map<int, std::string> getMaskPlaneDict() const;
        
        void printMaskPlanes() const;
        
//        virtual ~Mask();
        
private:

        MaskIVwPtrT _vwImagePtr;
        std::map<int, std::string> _maskPlaneDict;
        int _numPlanesUsed;
        lsst::mwi::data::DataProperty::PtrType _metaData;
        static const std::string maskPlanePrefix;
        unsigned int _offsetRows;
        unsigned int _offsetCols;
        
        void setOffsetRows(unsigned int offset);
        void setOffsetCols(unsigned int offset);
        
        MaskChannelT getBitMask(int plane) const { return 1 << plane; }

        int addMaskPlane(std::string name, int plane);
    };

}}  // lsst::fw
  
#ifndef SWIG // don't bother SWIG with .cc files
#include "Mask.cc"  
#endif
        
#endif // LSST_MASK_H


