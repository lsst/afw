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

#include "boost/cstdint.hpp"
#include "boost/shared_ptr.hpp"

#include "lsst/daf/base.h"
#include "lsst/daf/data/LsstBase.h"
#include "lsst/pex/exceptions.h"
#include "lsst/daf/base/Persistable.h"
//#include "lsst/afw/formatters/ImageFormatter.h"
//#include "lsst/afw/image/ImageExceptions.h"
#include "lsst/gil/Image.h"

namespace lsst {
namespace afw {
    namespace formatters {
        class MaskFormatter;
    }
namespace image {
    // all masks will initially be instantiated with the same pixel type
    typedef boost::uint16_t MaskPixel;

    template<typename MaskPixelT>
    class Mask : public ImageBase<MaskPixelT> {
    public:
        typedef boost::shared_ptr<Mask> Ptr;
        typedef boost::shared_ptr<const Mask> ConstPtr;
        typedef std::map<std::string, int> MaskPlaneDict;
        
        typedef typename ImageBase<MaskPixelT>::x_iterator x_iterator;
        
        // Constructors        
        explicit Mask(int nCols=0, int nRows=0, MaskPlaneDict const& planeDefs = MaskPlaneDict());
        explicit Mask(const std::pair<int, int> dimensions, MaskPlaneDict const& planeDefs = MaskPlaneDict());
        explicit Mask(std::string const& fileName, bool conformMasks=false, int hdu=0);

        Mask(const Mask& src, const bool deep=false);
        Mask(const Mask& src, const Bbox& bbox, const bool deep=false);
        
        // Operators

        Mask& operator=(MaskPixelT const rhs);
        Mask& operator=(const Mask& rhs);

        void operator|=(Mask const& rhs);
        void operator|=(MaskPixelT const rhs);

        void operator&=(Mask const& rhs);
        void operator&=(MaskPixelT const rhs);

        typename ImageBase<MaskPixelT>::PixelReference operator()(int x, int y);
        typename ImageBase<MaskPixelT>::PixelConstReference operator()(int x, int y) const;
        bool operator()(int x, int y, int plane) const;

        // I/O and FITS metadata
        
        //void readFits(const std::string& fileName, bool conformMasks=false, int hdu=0); // replaced by constructor
        void writeFits(std::string const& fileName) const;
        
        lsst::daf::base::DataProperty::PtrType getMetaData();

        // Mask Plane ops
        
        void clearAllMaskPlanes();
        void clearMaskPlane(int plane);

        //void setMaskPlaneValues(int plane, std::list<PixelCoord> pixelList);

        void setMaskPlaneValues(const int plane, const int x0, const int x1, const int y);

        //void setMaskPlaneValues(int plane, MaskPixelBooleanFunc<MaskPixelT> selectionFunc);
        
        static MaskPlaneDict parseMaskPlaneMetaData(lsst::daf::base::DataProperty::PtrType const);
        
        //int countMask(MaskPixelBooleanFunc<MaskPixelT>& testFunc, const vw::BBox2i maskRegion) const;
        
        //
        // Operations on the mask plane dictionary
        //
        void clearMaskPlaneDict();
        static int addMaskPlane(const std::string& name);
        void removeMaskPlane(const std::string& name);
        
        static int getMaskPlane(const std::string& name);
        static MaskPixelT getPlaneBitMask(const std::string& name);

        static int getNumPlanesMax()  { return 8*sizeof(MaskPixelT); }
        static int getNumPlanesUsed() { return _maskPlaneDict.size(); }
        static const MaskPlaneDict& getMaskPlaneDict() { return _maskPlaneDict; }
        static void printMaskPlanes();

        static void addMaskPlanesToMetaData(lsst::daf::base::DataProperty::PtrType);
        //
        // This one isn't static, it fixes up a given Mask's planes
        void conformMaskPlanes(MaskPlaneDict& masterPlaneDict);
        
        // Getters
#if 1                                   // Old name for boost::shared_ptrs
        lsst::daf::base::DataProperty::PtrType getMetaData() const { return _metaData; }
#else
        lsst::daf::base::DataProperty::Ptr      getMetaData()       { return _metaData; }
        lsst::daf::base::DataProperty::ConstPtr getMetaData() const { return _metaData; }
#endif
        
private:
        //LSST_PERSIST_FORMATTER(lsst::afw::formatters::MaskFormatter);
#if 1                                   // Old name for boost::shared_ptrs
        lsst::daf::base::DataProperty::PtrType _metaData;
#else
        lsst::daf::base::DataProperty::Ptr _metaData;
#endif
        int _myMaskDictVersion;         // version number for bitplane dictionary for this Mask

        static MaskPlaneDict _maskPlaneDict;
        static const std::string maskPlanePrefix;
        
        static int addMaskPlane(std::string name, int plane);

        static int getMaskPlaneNoThrow(const std::string& name);
        static MaskPixelT getBitMaskNoThrow(int plane);
        static MaskPixelT getBitMask(int plane);

        static int _maskDictVersion;    // version number for bitplane dictionary
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
    private:
        //
        // Make names in templatized base class visible (Meyers, Effective C++, Item 43)
        //
        using ImageBase<MaskPixelT>::_getRawView;
        using ImageBase<MaskPixelT>::_getRawImagePtr;
        using ImageBase<MaskPixelT>::_setRawView;
    };

}}}  // lsst::afw::image
        
#endif // LSST_AFW_IMAGE_MASK_H
