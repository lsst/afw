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

/*
 * LSST bitmasks
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
 * Represent a 2-dimensional array of bitmask pixels
 *
 * Some mask planes are always defined (although you can add more with Mask::addMaskPlane):
 *
 * - `BAD` This pixel is known to be bad (e.g. the amplifier is not working)
 * - `CR` This pixel is contaminated by a cosmic ray
 * - `DETECTED` This pixel lies within an object's Footprint
 * - `DETECTED_NEGATIVE` This pixel lies within an object's Footprint, and the detection was looking for
 *   pixels *below* a specified level
 * - `EDGE` This pixel is too close to the edge to be processed properly
 * - `INTRP` This pixel has been interpolated over @note should be called `INTERPOLATED`
 * - `SAT` This pixel is saturated and has bloomed @note should be called `SATURATED`
 * - `SUSPECT` This pixel is untrustworthy, and you may wish to discard any Source containing it
 */
template<typename MaskPixelT=lsst::afw::image::MaskPixel>
class Mask : public ImageBase<MaskPixelT> {
public:
    typedef std::shared_ptr<Mask> Ptr;
    typedef std::shared_ptr<const Mask> ConstPtr;
    typedef detail::MaskPlaneDict MaskPlaneDict;

    typedef detail::Mask_tag image_category;

    /// A templated class to return this classes' type (present in Image/Mask/MaskedImage)
    template<typename MaskPT=MaskPixelT>
    struct ImageTypeFactory {
        /// Return the desired type
        typedef Mask<MaskPT> type;
    };

    // Constructors
    /**
     * Construct a Mask initialized to 0x0
     *
     * @param width number of columns
     * @param height number of rows
     * @param planeDefs desired mask planes
     */
    explicit Mask(
        unsigned int width, unsigned int height,
        MaskPlaneDict const& planeDefs=MaskPlaneDict()
    );
    /**
     * Construct a Mask initialized to a specified value
     *
     * @param width number of columns
     * @param height number of rows
     * @param initialValue Initial value
     * @param planeDefs desired mask planes
     */
    explicit Mask(
        unsigned int width, unsigned int height,
        MaskPixelT initialValue,
        MaskPlaneDict const& planeDefs=MaskPlaneDict()
    );
    /**
     * Construct a Mask initialized to 0x0
     *
     * @param dimensions Number of columns, rows
     * @param planeDefs desired mask planes
     */
    explicit Mask(
        geom::Extent2I const & dimensions=geom::Extent2I(),
        MaskPlaneDict const& planeDefs=MaskPlaneDict()
    );
    /**
     * Construct a Mask initialized to a specified value
     *
     * @param dimensions Number of columns, rows
     * @param initialValue Initial value
     * @param planeDefs desired mask planes
     */
    explicit Mask(
        geom::Extent2I const & dimensions,
        MaskPixelT initialValue,
        MaskPlaneDict const& planeDefs=MaskPlaneDict()
    );
    /**
     * Construct a Mask initialized to 0x0
     *
     * @param bbox Desired number of columns/rows and origin
     * @param planeDefs desired mask planes
     */
    explicit Mask(geom::Box2I const & bbox,
                  MaskPlaneDict const& planeDefs=MaskPlaneDict());
    /**
     * Construct a Mask initialized to a specified value
     *
     * @param bbox Desired number of columns/rows and origin
     * @param initialValue Initial value
     * @param planeDefs desired mask planes
     */
    explicit Mask(geom::Box2I const & bbox, MaskPixelT initialValue,
                  MaskPlaneDict const& planeDefs=MaskPlaneDict());

    /**
     *  Construct a Mask by reading a regular FITS file.
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
     *  Construct a Mask by reading a FITS image in memory.
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
     *  Construct a Mask from an already-open FITS object.
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

    /**
     * Construct a Mask from another Mask
     *
     * @param src mask to copy
     * @param deep deep copy? (construct a view with shared pixels if false)
     */
    Mask(const Mask& src, const bool deep=false);
    /**
     * Construct a Mask from a subregion of another Mask
     *
     * @param src mask to copy
     * @param bbox subregion to copy
     * @param origin coordinate system of the bbox
     * @param deep deep copy? (construct a view with shared pixels if false)
     */
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

    /// OR a Mask into a Mask
    Mask& operator|=(Mask const& rhs);
    /// OR a bitmask into a Mask
    Mask& operator|=(MaskPixelT const rhs);

    /// AND a Mask into a Mask
    Mask& operator&=(Mask const& rhs);
    /// AND a bitmask into a Mask
    Mask& operator&=(MaskPixelT const rhs);
    /**
     * Return the bitmask corresponding to a vector of plane names OR'd together
     *
     * @throws lsst::pex::exceptions::InvalidParameterError if plane is invalid
     */
        static MaskPixelT getPlaneBitMask(const std::vector<std::string> &names);

    /// XOR a Mask into a Mask
    Mask& operator^=(Mask const& rhs);
    /// XOR a bitmask into a Mask
    Mask& operator^=(MaskPixelT const rhs);

    /**
     * get a reference to the specified pixel
     *
     * @param x x index
     * @param y y index
     */
    typename ImageBase<MaskPixelT>::PixelReference operator()(int x, int y);
    /**
     * get the specified pixel (const version)
     *
     * @param x x index
     * @param y y index
     */
    typename ImageBase<MaskPixelT>::PixelConstReference operator()(int x, int y) const;
    /**
     * is the specified mask plane set in the specified pixel?
     *
     * @param x x index
     * @param y y index
     * @param plane plane ID
     */
    bool operator()(int x, int y, int plane) const;
    /**
     * get a reference to the specified pixel checking array bounds
     *
     * @param x x index
     * @param y y index
     * @param check Check array bounds?
     */
    typename ImageBase<MaskPixelT>::PixelReference operator()(int x, int y, CheckIndices const& check);
    /**
     * get the specified pixel with array checking (const version)
     *
     * @param x x index
     * @param y y index
     * @param check Check array bounds?
     */
    typename ImageBase<MaskPixelT>::PixelConstReference operator()(int x, int y,
                                                                   CheckIndices const& check) const;
    /**
     * is the specified mask plane set in the specified pixel, checking array bounds?
     *
     * @param x x index
     * @param y y index
     * @param plane plane ID
     * @param check Check array bounds?
     */
    bool operator()(int x, int y, int plane, CheckIndices const& check) const;

    /**
     *  Write a mask to a regular FITS file.
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
     *  Write a mask to a FITS RAM file.
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
     *  Write a mask to an open FITS file object.
     *
     *  @param[in] fitsfile      A FITS file already open to the desired HDU.
     *  @param[in] metadata      Additional values to write to the header (may be null).
     */
    void writeFits(
        fits::Fits & fitsfile,
        CONST_PTR(lsst::daf::base::PropertySet) metadata=CONST_PTR(lsst::daf::base::PropertySet)()
    ) const;

    /**
     *  Read a Mask from a regular FITS file.
     *
     *  @param[in] filename    Name of the file to read.
     *  @param[in] hdu         Number of the "header-data unit" to read (where 0 is the Primary HDU).
     *                         The default value of INT_MIN is interpreted as "the first HDU with NAXIS != 0".
     */
    static Mask readFits(std::string const & filename, int hdu=INT_MIN) {
        return Mask<MaskPixelT>(filename, hdu);
    }

    /**
     *  Read a Mask from a FITS RAM file.
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

    /// Clear all the pixels
    void clearAllMaskPlanes();
    /// Clear the specified bit in all pixels
    void clearMaskPlane(int plane);
    /**
     * Set the bit specified by "planeId" for pixels (x0, y) ... (x1, y)
     */
    void setMaskPlaneValues(const int plane, const int x0, const int x1, const int y);
    /**
     * Given a PropertySet that contains the MaskPlane assignments, setup the MaskPlanes.
     *
     * @param metadata metadata from a Mask
     * @returns a dictionary of mask plane name: plane ID
     */
    static MaskPlaneDict parseMaskPlaneMetadata(CONST_PTR(lsst::daf::base::PropertySet) metadata);
    //
    // Operations on the mask plane dictionary
    //
    /// Reset the maskPlane dictionary
    static void clearMaskPlaneDict();
    static int addMaskPlane(const std::string& name);
    static void removeMaskPlane(const std::string& name);
    /**
     * @brief Clear all pixels of the specified mask and remove the plane from the mask plane dictionary;
     * optionally remove the plane from the default dictionary too.
     *
     * @param name of maskplane
     * @param removeFromDefault remove from default mask plane dictionary too
     *
     * @throws lsst::pex::exceptions::InvalidParameterError if plane is invalid
     */
    void removeAndClearMaskPlane(const std::string& name, bool const removeFromDefault=false);

    /**
     * Return the mask plane number corresponding to a plane name
     *
     * @throws lsst::pex::exceptions::InvalidParameterError if plane is invalid
     */
    static int getMaskPlane(const std::string& name);
    /**
     * Return the bitmask corresponding to a plane name
     *
     * @throws lsst::pex::exceptions::InvalidParameterError if plane is invalid
     */
    static MaskPixelT getPlaneBitMask(const std::string& name);

    static int getNumPlanesMax()  { return 8*sizeof(MaskPixelT); }
    static int getNumPlanesUsed();
    /**
     * Return the Mask's maskPlaneDict
     */
    MaskPlaneDict const& getMaskPlaneDict() const;
    /// print the mask plane dictionary to std::cout
    void printMaskPlanes() const;

    /**
     * Given a PropertySet, replace any existing MaskPlane assignments with the current ones.
     */
    static void addMaskPlanesToMetadata(PTR(lsst::daf::base::PropertySet));
    //
    // This one isn't static, it fixes up a given Mask's planes
    /**
     * Adjust this mask to conform to the standard Mask class's mask plane dictionary,
     * adding any new mask planes to the standard.
     *
     * Ensures that this mask (presumably from some external source) has the same plane assignments
     * as the Mask class. If a change in plane assignments is needed, the bits within each pixel
     * are permuted as required.  The provided `masterPlaneDict` describes the true state of the bits
     * in this Mask's pixels and overrides its current MaskDict
     *
     * Any new mask planes found in this mask are added to unused slots in the Mask class's mask plane dictionary.
     *
     * @param masterPlaneDict mask plane dictionary currently in use for this mask
     */
    void conformMaskPlanes(const MaskPlaneDict& masterPlaneDict);

private:
    //LSST_PERSIST_FORMATTER(lsst::afw::formatters::MaskFormatter)
    PTR(detail::MaskDict) _maskDict;    // our bitplane dictionary

    static PTR(detail::MaskDict) _maskPlaneDict();
    static int _setMaskPlaneDict(MaskPlaneDict const& mpd);
    static const std::string maskPlanePrefix;

    /**
     * set the name of a mask plane, with minimal checking.
     *
     * This is a private function and is mainly used by setMaskPlaneMetadata
     *
     * @param name new name of mask plane
     * @param plane ID of mask plane to be (re)named
     */
    static int addMaskPlane(std::string name, int plane);

    /// Return the mask plane number corresponding to a plane name, or -1 if not found
    static int getMaskPlaneNoThrow(const std::string& name);
    /// Return the bitmask corresponding to a plane ID, or 0 if invalid
    static MaskPixelT getBitMaskNoThrow(int plane);
    /**
     * Return the bitmask corresponding to plane ID
     *
     * @throws lsst::pex::exceptions::InvalidParameterError if plane is invalid
     */
    static MaskPixelT getBitMask(int plane);

    /**
     * Initialise mask planes; called by constructors
     */
    void _initializePlanes(MaskPlaneDict const& planeDefs); // called by ctors

    //
    // Make names in templatized base class visible (Meyers, Effective C++, Item 43)
    //
    using ImageBase<MaskPixelT>::_getRawView;
    using ImageBase<MaskPixelT>::swap;

    /**
     * Check that masks have the same dictionary version
     *
     * @throws lsst::pex::exceptions::RuntimeError
     */
    void checkMaskDictionaries(Mask const& other);
};

template<typename PixelT>
void swap(Mask<PixelT>& a, Mask<PixelT>& b);

}}}  // lsst::afw::image

#endif // LSST_AFW_IMAGE_MASK_H
