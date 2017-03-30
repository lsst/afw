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

#include <memory>

#include "lsst/base.h"
#include "lsst/daf/base/Citizen.h"
#include "lsst/daf/base/Persistable.h"
#include "lsst/daf/base/PropertySet.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/formatters/ImageFormatter.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/LsstImageTypes.h"

namespace lsst {
namespace afw {
    namespace formatters {
        template<typename> class MaskFormatter;
    }
namespace image {

namespace detail {
    class MaskDict;                     // forward declaration
}

// all masks will initially be instantiated with the same pixel type
namespace detail {
    /// tag for a Mask
    struct Mask_tag : public detail::basic_tag { };

    typedef std::map<std::string, int> MaskPlaneDict;
}

/**
 * \brief Represent a 2-dimensional array of bitmask pixels
 *
 * Some mask planes are always defined (although you can add more with Mask::addMaskPlane):
 *
 <UL>
 <LI> \c BAD This pixel is known to be bad (e.g. the amplifier is not working)

 <LI> \c CR This pixel is contaminated by a cosmic ray

 <LI> \c DETECTED This pixel lies within an object's Footprint

 <LI> \c DETECTED_NEGATIVE This pixel lies within an object's Footprint, and the detection was looking for
 pixels \em below a specified level

 <LI> \c EDGE This pixel is too close to the edge to be processed properly

 <LI> \c INTRP This pixel has been interpolated over \note should be called \c INTERPOLATED

 <LI> \c SAT This pixel is saturated and has bloomed \note should be called \c SATURATED

 <LI> \c SUSPECT This pixel is untrustworthy, and you may wish to discard any Source containing it
 </UL>
 */
template<typename MaskPixelT=lsst::afw::image::MaskPixel>
class Mask : public ImageBase<MaskPixelT> {
public:
    typedef std::shared_ptr<Mask> Ptr;
    typedef std::shared_ptr<const Mask> ConstPtr;
    typedef detail::MaskPlaneDict MaskPlaneDict;

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
    explicit Mask(
        unsigned int width, unsigned int height,
        MaskPlaneDict const& planeDefs=MaskPlaneDict()
    );
    explicit Mask(
        unsigned int width, unsigned int height,
        MaskPixelT initialValue,
        MaskPlaneDict const& planeDefs=MaskPlaneDict()
    );
    explicit Mask(
        geom::Extent2I const & dimensions=geom::Extent2I(),
        MaskPlaneDict const& planeDefs=MaskPlaneDict()
    );
    explicit Mask(
        geom::Extent2I const & dimensions,
        MaskPixelT initialValue,
        MaskPlaneDict const& planeDefs=MaskPlaneDict()
    );
    explicit Mask(geom::Box2I const & bbox,
                  MaskPlaneDict const& planeDefs=MaskPlaneDict());
    explicit Mask(geom::Box2I const & bbox, MaskPixelT initialValue,
                  MaskPlaneDict const& planeDefs=MaskPlaneDict());

    /**
     *  @brief Construct a Mask by reading a regular FITS file.
     *
     *  @param[in]      fileName      File to read.
     *  @param[in]      hdu           HDU to read, 0-indexed (i.e. 0=Primary HDU).  The special value
     *                                of INT_MIN reads the Primary HDU unless it is empty, in which case it
     *                                reads the first extension HDU.
     *  @param[in,out]  metadata      Metadata read from the header (may be null).
     *  @param[in]      bbox          If non-empty, read only the pixels within the bounding box.
     *  @param[in]      origin        Coordinate system of the bounding box; if PARENT, the bounding box
     *                                should take into account the xy0 saved with the image.
     *  @param[in]      conformMasks  If true, make Mask conform to the mask layout in the file.
     *
     *  The meaning of the bitplanes is given in the header.  If conformMasks is false (default),
     *  the bitvalues will be changed to match those in Mask's plane dictionary.  If it's true, the
     *  bitvalues will be left alone, but Mask's dictionary will be modified to match the
     *  on-disk version.
     */
    explicit Mask(
        std::string const & fileName, int hdu=INT_MIN,
        PTR(lsst::daf::base::PropertySet) metadata=PTR(lsst::daf::base::PropertySet)(),
        geom::Box2I const & bbox=geom::Box2I(),
        ImageOrigin origin=PARENT,
        bool conformMasks=false
    );

    /**
     *  @brief Construct a Mask by reading a FITS image in memory.
     *
     *  @param[in]      manager       An object that manages the memory buffer to read.
     *  @param[in]      hdu           HDU to read, 0-indexed (i.e. 0=Primary HDU).  The special value
     *                                of INT_MIN reads the Primary HDU unless it is empty, in which case it
     *                                reads the first extension HDU.
     *  @param[in,out]  metadata      Metadata read from the header (may be null).
     *  @param[in]      bbox          If non-empty, read only the pixels within the bounding box.
     *  @param[in]      origin        Coordinate system of the bounding box; if PARENT, the bounding box
     *                                should take into account the xy0 saved with the image.
     *  @param[in]      conformMasks  If true, make Mask conform to the mask layout in the file.
     *
     *  The meaning of the bitplanes is given in the header.  If conformMasks is false (default),
     *  the bitvalues will be changed to match those in Mask's plane dictionary.  If it's true, the
     *  bitvalues will be left alone, but Mask's dictionary will be modified to match the
     *  on-disk version.
     */
    explicit Mask(
        fits::MemFileManager & manager, int hdu=INT_MIN,
        PTR(lsst::daf::base::PropertySet) metadata=PTR(lsst::daf::base::PropertySet)(),
        geom::Box2I const & bbox=geom::Box2I(),
        ImageOrigin origin=PARENT,
        bool conformMasks=false
    );

    /**
     *  @brief Construct a Mask from an already-open FITS object.
     *
     *  @param[in]      fitsfile      A FITS object to read from, already at the desired HDU.
     *  @param[in,out]  metadata      Metadata read from the header (may be null).
     *  @param[in]      bbox          If non-empty, read only the pixels within the bounding box.
     *  @param[in]      origin        Coordinate system of the bounding box; if PARENT, the bounding box
     *                                should take into account the xy0 saved with the image.
     *  @param[in]      conformMasks  If true, make Mask conform to the mask layout in the file.
     *
     *  The meaning of the bitplanes is given in the header.  If conformMasks is false (default),
     *  the bitvalues will be changed to match those in Mask's plane dictionary.  If it's true, the
     *  bitvalues will be left alone, but Mask's dictionary will be modified to match the
     *  on-disk version.
     */
    explicit Mask(
        fits::Fits & fitsfile,
        PTR(lsst::daf::base::PropertySet) metadata=PTR(lsst::daf::base::PropertySet)(),
        geom::Box2I const & bbox=geom::Box2I(),
        ImageOrigin origin=PARENT,
        bool conformMasks=false
    );

    // generalised copy constructor
    template<typename OtherPixelT>
    Mask(Mask<OtherPixelT> const& rhs, const bool deep) :
        image::ImageBase<MaskPixelT>(rhs, deep),
        _maskDict(rhs._maskDict) {}

    Mask(const Mask& src, const bool deep=false);
    Mask(
        const Mask& src,
        const geom::Box2I & bbox,
        ImageOrigin const origin=PARENT,
        const bool deep=false
    );

    explicit Mask(ndarray::Array<MaskPixelT,2,1> const & array, bool deep=false,
                  geom::Point2I const & xy0=geom::Point2I());

    void swap(Mask& rhs);
    // Operators

    Mask& operator=(MaskPixelT const rhs);
    Mask& operator=(const Mask& rhs);

    Mask& operator|=(Mask const& rhs);
    Mask& operator|=(MaskPixelT const rhs);

    Mask& operator&=(Mask const& rhs);
    Mask& operator&=(MaskPixelT const rhs);
        static MaskPixelT getPlaneBitMask(const std::vector<std::string> &names);

    Mask& operator^=(Mask const& rhs);
    Mask& operator^=(MaskPixelT const rhs);

    typename ImageBase<MaskPixelT>::PixelReference operator()(int x, int y);
    typename ImageBase<MaskPixelT>::PixelConstReference operator()(int x, int y) const;
    bool operator()(int x, int y, int plane) const;
    typename ImageBase<MaskPixelT>::PixelReference operator()(int x, int y, CheckIndices const&);
    typename ImageBase<MaskPixelT>::PixelConstReference operator()(int x, int y,
                                                                   CheckIndices const&) const;
    bool operator()(int x, int y, int plane, CheckIndices const&) const;

    /**
     *  @brief Write a mask to a regular FITS file.
     *
     *  @param[in] fileName      Name of the file to write.
     *  @param[in] metadata      Additional values to write to the header (may be null).
     *  @param[in] mode          "w"=Create a new file; "a"=Append a new HDU.
     */
    void writeFits(
        std::string const& fileName,
        CONST_PTR(lsst::daf::base::PropertySet) metadata=PTR(lsst::daf::base::PropertySet)(),
        std::string const& mode="w"
    ) const;

    /**
     *  @brief Write a mask to a FITS RAM file.
     *
     *  @param[in] manager       Manager object for the memory block to write to.
     *  @param[in] metadata      Additional values to write to the header (may be null).
     *  @param[in] mode          "w"=Create a new file; "a"=Append a new HDU.
     */
    void writeFits(
        fits::MemFileManager & manager,
        CONST_PTR(lsst::daf::base::PropertySet) metadata=PTR(lsst::daf::base::PropertySet)(),
        std::string const& mode="w"
    ) const;

    /**
     *  @brief Write a mask to an open FITS file object.
     *
     *  @param[in] fitsfile      A FITS file already open to the desired HDU.
     *  @param[in] metadata      Additional values to write to the header (may be null).
     */
    void writeFits(
        fits::Fits & fitsfile,
        CONST_PTR(lsst::daf::base::PropertySet) metadata=CONST_PTR(lsst::daf::base::PropertySet)()
    ) const;

    /**
     *  @brief Read a Mask from a regular FITS file.
     *
     *  @param[in] filename    Name of the file to read.
     *  @param[in] hdu         Number of the "header-data unit" to read (where 0 is the Primary HDU).
     *                         The default value of INT_MIN is interpreted as "the first HDU with NAXIS != 0".
     */
    static Mask readFits(std::string const & filename, int hdu=INT_MIN) {
        return Mask<MaskPixelT>(filename, hdu);
    }

    /**
     *  @brief Read a Mask from a FITS RAM file.
     *
     *  @param[in] manager     Object that manages the memory to be read.
     *  @param[in] hdu         Number of the "header-data unit" to read (where 0 is the Primary HDU).
     *                         The default value of INT_MIN is interpreted as "the first HDU with NAXIS != 0".
     */
    static Mask readFits(fits::MemFileManager & manager, int hdu=INT_MIN) {
        return Mask<MaskPixelT>(manager, hdu);
    }

    /// Interpret a mask value as a comma-separated list of mask plane names
    static std::string interpret(MaskPixelT value);
    std::string getAsString(int x, int y) { return interpret((*this)(x, y)); }

    // Mask Plane ops

    void clearAllMaskPlanes();
    void clearMaskPlane(int plane);
    void setMaskPlaneValues(const int plane, const int x0, const int x1, const int y);
    static MaskPlaneDict parseMaskPlaneMetadata(CONST_PTR(lsst::daf::base::PropertySet));
    //
    // Operations on the mask plane dictionary
    //
    static void clearMaskPlaneDict();
    static int addMaskPlane(const std::string& name);
    static void removeMaskPlane(const std::string& name);
    void removeAndClearMaskPlane(const std::string& name, bool const removeFromDefault=false);

    static int getMaskPlane(const std::string& name);
    static MaskPixelT getPlaneBitMask(const std::string& name);

    static int getNumPlanesMax()  { return 8*sizeof(MaskPixelT); }
    static int getNumPlanesUsed();
    MaskPlaneDict const& getMaskPlaneDict() const;
    void printMaskPlanes() const;

    static void addMaskPlanesToMetadata(PTR(lsst::daf::base::PropertySet));
    //
    // This one isn't static, it fixes up a given Mask's planes
    void conformMaskPlanes(const MaskPlaneDict& masterPlaneDict);

private:
    //LSST_PERSIST_FORMATTER(lsst::afw::formatters::MaskFormatter)
    PTR(detail::MaskDict) _maskDict;    // our bitplane dictionary

    static PTR(detail::MaskDict) _maskPlaneDict();
    static int _setMaskPlaneDict(MaskPlaneDict const& mpd);
    static const std::string maskPlanePrefix;

    static int addMaskPlane(std::string name, int plane);

    static int getMaskPlaneNoThrow(const std::string& name);
    static MaskPixelT getBitMaskNoThrow(int plane);
    static MaskPixelT getBitMask(int plane);

    void _initializePlanes(MaskPlaneDict const& planeDefs); // called by ctors

    //
    // Make names in templatized base class visible (Meyers, Effective C++, Item 43)
    //
    using ImageBase<MaskPixelT>::_getRawView;
    using ImageBase<MaskPixelT>::swap;

    void checkMaskDictionaries(Mask const& other);
};

template<typename PixelT>
void swap(Mask<PixelT>& a, Mask<PixelT>& b);

}}}  // lsst::afw::image

#endif // LSST_AFW_IMAGE_MASK_H
