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
#include "lsst/mwi/data/LsstBase.h"
#include "lsst/mwi/exceptions/Exception.h"
#include "lsst/mwi/data/DataProperty.h"
#include "lsst/mwi/utils/Trace.h"
#include "lsst/fw/LSSTFitsResource.h"


namespace lsst {

    namespace fw {
        using namespace vw;
        using namespace std;
        
        using lsst::mwi::data::LsstBase;
        using lsst::mwi::data::DataPropertyPtrT;
        using namespace lsst::mwi::exceptions;

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
        class Mask : private LsstBase {
        public:
            typedef typename PixelChannelType<MaskPixelT>::type MaskChannelT;
            typedef ImageView<MaskPixelT> MaskIVwT;
            typedef boost::shared_ptr<Mask<MaskPixelT> > MaskPtrT;
            typedef boost::shared_ptr<MaskIVwT> MaskIVwPtrT;
            typedef typename vw::ImageView<MaskPixelT>::pixel_accessor pixel_accessor;
            
            Mask();
            
            Mask(MaskIVwPtrT image);
            
            Mask(int nCols, int nRows);
            
            void readFits(const string& fileName, int hdu=0);
            
            void writeFits(const string& fileName);
            
            DataPropertyPtrT getMetaData();
            
            int addMaskPlane(const string& name);
            
            void removeMaskPlane(const string& name);
            
#if !defined(SWIG)
            void getMaskPlane(const string& name, int& plane) const;
#endif
            int getMaskPlane(const string& name) const;
            
            bool getPlaneBitMask(const string& name,
                                 MaskChannelT& bitMask) const;
            MaskChannelT getPlaneBitMask(const string& name) const;
            
            void clearMaskPlane(int plane);
            
            void clearAllMaskPlanes();
            
            void setMaskPlaneValues(int plane, list<PixelCoord> pixelList);
            void setMaskPlaneValues(const int plane, const int x0, const int x1, const int y);            
            void setMaskPlaneValues(int plane, MaskPixelBooleanFunc<MaskPixelT> selectionFunc);
            
            void parseMaskPlaneMetaData(const DataPropertyPtrT);
            
            void addMaskPlaneMetaData(DataPropertyPtrT);
            
            int countMask(MaskPixelBooleanFunc<MaskPixelT>& testFunc,
                          const BBox2i maskRegion) const;
            
            MaskPtrT getSubMask(const BBox2i maskRegion) const;
            
            void replaceSubMask(const BBox2i maskRegion, MaskPtrT insertMask);
            
            pixel_accessor origin() { return getIVwPtr()->origin(); }
            
            MaskChannelT operator ()(int x, int y) const;
            
            bool operator ()(int x, int y, int plane) const;
            
            Mask<MaskPixelT>& operator |= (const Mask<MaskPixelT>& inputMask);
            
            unsigned int getCols() const { return _image.cols(); }
            unsigned int getRows() const { return _image.rows(); }
            unsigned int getOffsetCols() const { return _offsetCols; }
            unsigned int getOffsetRows() const { return _offsetRows; }


            MaskIVwPtrT getIVwPtr() const {
                return _imagePtr; // did this increment reference count or not....and does this violate const??
            }                

            MaskIVwT& getIVw() const { return _image; }

            map<int, std::string> getMaskPlaneDict() const;
            
            void printMaskPlanes() const;
            
//         virtual ~Mask();
            
    private:
            MaskIVwPtrT _imagePtr;
            MaskIVwT& _image;
            map<int, std::string> _maskPlaneDict;
            const int _numPlanesMax;
            int _numPlanesUsed;
            MaskChannelT _planeBitMask[8 * sizeof(MaskChannelT)];
            MaskChannelT _planeBitMaskComplemented[8 * sizeof(MaskChannelT)];
            DataPropertyPtrT _metaData;
            static const std::string maskPlanePrefix;
            unsigned int _offsetRows;
            unsigned int _offsetCols;
            
            void setOffsetRows(unsigned int offset);
            void setOffsetCols(unsigned int offset);

            int addMaskPlane(string name, int plane);
        };
  
#include "Mask.cc"  
        
        
    } // namespace fw

} // namespace lsst

#endif // LSST_MASK_H


