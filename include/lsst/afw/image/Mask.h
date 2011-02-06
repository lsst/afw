// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
/**
 * \file
 * \brief LSST bitmasks
 */

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
#include "lsst/afw/formatters/ImageFormatter.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/LsstImageTypes.h"

namespace lsst {
namespace afw {
    namespace formatters {
        template<typename> class MaskFormatter;
    }
namespace image {
    // all masks will initially be instantiated with the same pixel type
    namespace detail {
        /// tag for a Mask
        struct Mask_tag : public detail::basic_tag { };
    }
    /// Represent a 2-dimensional array of bitmask pixels
    template<typename MaskPixelT=lsst::afw::image::MaskPixel>
    class Mask : public ImageBase<MaskPixelT> {
    public:
        typedef boost::shared_ptr<Mask> Ptr;
        typedef boost::shared_ptr<const Mask> ConstPtr;
        typedef std::map<std::string, int> MaskPlaneDict;
        
        typedef detail::Mask_tag image_category;

#if !defined(SWIG)
        /// A templated class to return this classes' type (present in Image/Mask/MaskedImage)
        template<typename MaskPT=MaskPixelT>
        struct ImageTypeFactory {
            /// Return the desired type
            typedef Mask<MaskPT> type;
        };
#endif

        // Constructors        
        explicit Mask(int nCols=0, int nRows=0, MaskPlaneDict const& planeDefs = MaskPlaneDict());
        explicit Mask(int nCols, int nRows, MaskPixelT initialValue, MaskPlaneDict const& planeDefs = MaskPlaneDict());
        explicit Mask(const std::pair<int, int> dimensions,
                      MaskPlaneDict const& planeDefs = MaskPlaneDict());
        explicit Mask(const std::pair<int, int> dimensions, MaskPixelT initialValue,
                      MaskPlaneDict const& planeDefs = MaskPlaneDict());
        explicit Mask(std::string const& fileName, int const hdu=0,
                      lsst::daf::base::PropertySet::Ptr metadata=lsst::daf::base::PropertySet::Ptr(),
                      BBox const& bbox=BBox(), bool const conformMasks=false
                     );           

        // generalised copy constructor
        template<typename OtherPixelT>
        Mask(Mask<OtherPixelT> const& rhs, const bool deep) :
            image::ImageBase<MaskPixelT>(rhs, deep) {}

        Mask(const Mask& src, const bool deep=false);
        Mask(const Mask& src, const BBox& bbox, const bool deep=false);
        
        void swap(Mask& rhs);
        // Operators

        Mask& operator=(MaskPixelT const rhs);
        Mask& operator=(const Mask& rhs);

        void operator|=(Mask const& rhs);
        void operator|=(MaskPixelT const rhs);

        void operator&=(Mask const& rhs);
        void operator&=(MaskPixelT const rhs);

        void operator^=(Mask const& rhs);
        void operator^=(MaskPixelT const rhs);

        typename ImageBase<MaskPixelT>::PixelReference operator()(int x, int y);
        typename ImageBase<MaskPixelT>::PixelConstReference operator()(int x, int y) const;
        bool operator()(int x, int y, int plane) const;
        typename ImageBase<MaskPixelT>::PixelReference operator()(int x, int y, CheckIndices const&);
        typename ImageBase<MaskPixelT>::PixelConstReference operator()(int x, int y,
                                                                       CheckIndices const&) const;
        bool operator()(int x, int y, int plane, CheckIndices const&) const;

        // I/O and FITS metadata
        
        //void readFits(const std::string& fileName, bool conformMasks=false, int hdu=0); // replaced by constructor
        void writeFits(std::string const& fileName,
                       boost::shared_ptr<const lsst::daf::base::PropertySet> metadata=
                                                                                   lsst::daf::base::PropertySet::Ptr(),
                       std::string const& mode="w"
                      ) const;
        
        // Mask Plane ops
        
        void clearAllMaskPlanes();
        void clearMaskPlane(int plane);
        void setMaskPlaneValues(const int plane, const int x0, const int x1, const int y);
        static MaskPlaneDict parseMaskPlaneMetadata(lsst::daf::base::PropertySet::Ptr const);
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

        static void addMaskPlanesToMetadata(lsst::daf::base::PropertySet::Ptr);
        //
        // This one isn't static, it fixes up a given Mask's planes
        void conformMaskPlanes(const MaskPlaneDict& masterPlaneDict);
        
        // Getters
        
private:
        //LSST_PERSIST_FORMATTER(lsst::afw::formatters::MaskFormatter)
        int _myMaskDictVersion;         // version number for bitplane dictionary for this Mask

        static MaskPlaneDict _maskPlaneDict;
        static const std::string maskPlanePrefix;
        
        static int addMaskPlane(std::string name, int plane);

        static int getMaskPlaneNoThrow(const std::string& name);
        static MaskPixelT getBitMaskNoThrow(int plane);
        static MaskPixelT getBitMask(int plane);

        static int _maskDictVersion;    // version number for bitplane dictionary

        void _initializePlanes(MaskPlaneDict const& planeDefs); // called by ctors
        //
        // Check that masks have the same dictionary version
        //
        // @throw lsst::pex::exceptions::Runtime
        //
        void checkMaskDictionaries(Mask const &other) const {
            if (_myMaskDictVersion != other._myMaskDictVersion) {
                throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                                  (boost::format("Mask dictionary versions do not match; %d v. %d") %
                                   _myMaskDictVersion % other._myMaskDictVersion).str());
            }
        }        
    private:
        //
        // Make names in templatized base class visible (Meyers, Effective C++, Item 43)
        //
        using ImageBase<MaskPixelT>::_getRawView;
        using ImageBase<MaskPixelT>::swap;
    };

    template<typename PixelT>
    void swap(Mask<PixelT>& a, Mask<PixelT>& b);
    
}}}  // lsst::afw::image
        
#endif // LSST_AFW_IMAGE_MASK_H
