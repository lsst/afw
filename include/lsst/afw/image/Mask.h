// -*- lsst-c++ -*-
///////////////////////////////////////////////////////////
//  Mask.h
//  Implementation of the Class Mask
//  Created on:      09-Feb-2007 15:57:46
//  Original author: Tim Axelrod
///////////////////////////////////////////////////////////

#ifndef LSST_AFW_IMAGE_MASK_H
#define LSST_AFW_IMAGE_MASK_H

#include <list>
#include <map>
#include <string>

#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>
#include <vw/Image.h>
#include <vw/Math/BBox.h>

#include <lsst/daf/data.h>
#include <lsst/pex/exceptions.h>
#include <lsst/daf/persistence/Persistable.h>
#include <lsst/pex/utils/Trace.h>
#include <lsst/afw/image/LSSTFitsResource.h>
#include <lsst/afw/image/ImageExceptions.h>

namespace lsst {
namespace afw {
namespace image {
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
    namespace formatters {
        template<typename MaskPixelT> class MaskFormatter;
    }
    
    template <typename MaskPixelT> class MaskPixelBooleanFunc {
    public:
        MaskPixelBooleanFunc(Mask<MaskPixelT>& m) : _mask(m) {}
        virtual bool operator () (MaskPixelT) const;
        virtual ~MaskPixelBooleanFunc() {}
    protected:
        Mask<MaskPixelT>& _mask;
    };
    
    template<typename MaskPixelT>
    class Mask : public lsst::daf::persitence::Persistable,
                 public lsst::daf::data::LsstBase {
    public:
        typedef typename PixelChannelType<MaskPixelT>::type MaskChannelT;
        typedef vw::ImageView<MaskPixelT> MaskIVwT;
        typedef boost::shared_ptr<Mask<MaskPixelT> > Ptr;
        typedef boost::shared_ptr<Mask<MaskPixelT> > MaskPtrT; // deprecated; use Ptr
        typedef boost::shared_ptr<MaskIVwT> MaskIVwPtrT;
        typedef typename vw::ImageView<MaskPixelT>::pixel_accessor pixel_accessor;
        typedef std::map<std::string, int> MaskPlaneDict;

        // Constructors        

        explicit Mask(MaskPlaneDict const& planeDefs = MaskPlaneDict());
        
        explicit Mask(MaskIVwPtrT vwImagePtr, MaskPlaneDict const& planeDefs = MaskPlaneDict());
        
        explicit Mask(int nCols, int nRows, MaskPlaneDict const& planeDefs = MaskPlaneDict());

        // Operators

        Mask<MaskPixelT>& operator=(const Mask<MaskPixelT>& rhs);

        Mask<MaskPixelT>& operator |= (const Mask<MaskPixelT>& inputMask);
        Mask<MaskPixelT>& operator |= (MaskPixelT const inputMask);

        Mask<MaskPixelT>& operator &= (MaskPixelT const opMask);

        MaskChannelT operator ()(int x, int y) const;
        
        bool operator ()(int x, int y, int plane) const;

        // I/O and FITS metadata
        
        void readFits(const std::string& fileName, bool conformMasks=false, int hdu=0);
        
        void writeFits(const std::string& fileName);
        
        lsst::daf::data::DataProperty::PtrType getMetaData();

        // Mask Plane ops
        
        int addMaskPlane(const std::string& name);
        
        void removeMaskPlane(const std::string& name);
        
#if !defined(SWIG)
        void getMaskPlane(const std::string& name, int& plane) const;
#endif
        int getMaskPlane(const std::string& name) const;
        
        bool getPlaneBitMask(const std::string& name,
                             MaskChannelT& bitMask) const;
        MaskChannelT getPlaneBitMask(const std::string& name) const;

        void clearMaskPlaneDict();
        
        void clearAllMaskPlanes();
        void clearMaskPlane(int plane);

        void setMaskPlaneValues(int plane, std::list<PixelCoord> pixelList);

        void setMaskPlaneValues(const int plane, const int x0, const int x1, const int y);

        void setMaskPlaneValues(int plane, MaskPixelBooleanFunc<MaskPixelT> selectionFunc);
        
        MaskPlaneDict parseMaskPlaneMetaData(lsst::daf::data::DataProperty::PtrType const) const;
        
        void addMaskPlaneMetaData(lsst::daf::data::DataProperty::PtrType);
        
        int countMask(MaskPixelBooleanFunc<MaskPixelT>& testFunc,
                      const vw::BBox2i maskRegion) const;
        
        int getNumPlanesMax() const { return 8 * sizeof(MaskChannelT); }

        int getNumPlanesUsed() const { return _maskPlaneDict.size(); }

        MaskPlaneDict getMaskPlaneDict() const;
        
        void printMaskPlanes() const;

        void conformMaskPlanes(MaskPlaneDict masterPlaneDict);

        // SubMask ops

        MaskPtrT getSubMask(const vw::BBox2i maskRegion) const;
        
        void replaceSubMask(const vw::BBox2i maskRegion, MaskPtrT insertMask);
        
        pixel_accessor origin() const { return getIVwPtr()->origin(); }
        
        // Getters

        unsigned int getCols() const { return _vwImagePtr->cols(); }
        unsigned int getRows() const { return _vwImagePtr->rows(); }
        unsigned int getOffsetCols() const { return _offsetCols; }
        unsigned int getOffsetRows() const { return _offsetRows; }

        MaskIVwPtrT getIVwPtr() const {
            return _vwImagePtr; // did this increment reference count or not....and does this violate const??
        }                

        MaskIVwT& getIVw() const { return *_vwImagePtr; }

        
private:
        LSST_PERSIST_FORMATTER(formatters::MaskFormatter<MaskPixelT>);

        MaskIVwPtrT _vwImagePtr;
        lsst::daf::data::DataProperty::PtrType _metaData;
        static MaskPlaneDict _maskPlaneDict;
        static const std::string maskPlanePrefix;
        unsigned int _offsetRows;
        unsigned int _offsetCols;
        
        void setOffsetRows(unsigned int offset);
        void setOffsetCols(unsigned int offset);
        
        MaskChannelT getBitMask(int plane) const;

        static int _maskDictVersion;    // version number for bitplane dictionary
        int _myMaskDictVersion;         // version number for bitplane dictionary for this Mask
        //
        // Check that masks have the same dictionary version
        //
        // @throw lsst::pex::exceptions::Runtime
        //
        void checkMaskDictionaries(Mask const &other) const {
            if (_myMaskDictVersion != other._myMaskDictVersion) {
                throw lsst::pex::exceptions::Runtime("Mask dictionary versions do not match");
            }
        }        

        int addMaskPlane(std::string name, int plane);
    };

}}}  // lsst::afw::image
        
#endif // LSST_AFW_IMAGE_MASK_H


